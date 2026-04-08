# -*- coding: utf-8 -*-
import os
import math
import torch
from e3nn import o3
import time

# ------------------- Wigner D 相关（保持不变） -------------------
_THIS_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
_JD_PATH = os.path.join(_THIS_DIR, "Jd.pt")
if os.path.exists(_JD_PATH):
    _Jd = torch.load(_JD_PATH)
else:
    raise RuntimeError(f"Missing Jd.pt at {_JD_PATH}")

def _now_and_sync(t: torch.Tensor):
    if t.is_cuda:
        torch.cuda.synchronize(t.device)
    return time.perf_counter()

def _z_rot_mat(angle, l):
    """
    构造围绕 z 轴的分块旋转矩阵（用于 Wigner-D 近似构造），保持原实现不变。
    angle: [*,] 张量
    l: 非负整数
    """
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M

def wigner_D(l, alpha, beta, gamma):
    """
    ZYZ 欧拉角参数化的 Wigner-D，保持原实现。
    """
    if not l < len(_Jd):
        raise NotImplementedError(f"wigner D maximum l implemented is {len(_Jd) - 1}")
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def compute_D(vectors: torch.Tensor, l_max: int):
    eps = 1e-11
    N, device, dtype = vectors.shape[0], vectors.device, vectors.dtype

    z = torch.tensor([0., 0., 1.], device=device, dtype=dtype).expand(N, 3)
    v_hat = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(eps)

    axis = torch.cross(v_hat, z, dim=1)                 # [N,3]
    s = axis.norm(dim=1, keepdim=True)                  # sin(theta)
    c = (v_hat * z).sum(dim=1, keepdim=True)            # cos(theta)
    a = axis / (s + eps)

    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    zero = torch.zeros_like(ax)
    K = torch.stack([
        torch.stack([zero, -az,  ay], dim=-1),
        torch.stack([ az,  zero, -ax], dim=-1),
        torch.stack([-ay,   ax,  zero], dim=-1),
    ], dim=1)
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N, 3, 3)
    aaT = a.unsqueeze(2) * a.unsqueeze(1)

    R_rod = c.view(N,1,1)*I + s.view(N,1,1)*K + (1.-c).view(N,1,1)*aaT


    s_thr = 5e-4
    tau = 1e-4
    w = torch.sigmoid((s_thr - s) / tau)                # [N,1]
    pos = (c >= 0).float()
    neg = 1.0 - pos
    R_pole = (pos.view(N,1,1) * I +
              neg.view(N,1,1) * torch.diag_embed(torch.tensor([1., -1., -1.], device=device, dtype=dtype)).expand(N,3,3))
    R = (1 - w).view(N,1,1) * R_rod + w.view(N,1,1) * R_pole


    Rt = R.transpose(1, 2)
    R11,R12,R13 = Rt[:,0,0],Rt[:,0,1],Rt[:,0,2]
    R21,R22,R23 = Rt[:,1,0],Rt[:,1,1],Rt[:,1,2]
    R31,R32,R33 = Rt[:,2,0],Rt[:,2,1],Rt[:,2,2]

    # beta = atan2( sqrt(R13^2 + R23^2), R33_clamped )
    r = torch.sqrt(R13*R13 + R23*R23 + 1e-12)  
    R33c = R33.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    beta  = torch.atan2(r, R33c)

    sinb = torch.sin(beta)  
    m = sinb > 1e-4

    alpha = torch.zeros_like(beta)
    gamma = torch.zeros_like(beta)

    alpha = torch.where(m, torch.atan2(R23, R13 + 1e-12), alpha)
    gamma = torch.where(m, torch.atan2(-R32, -R31 + 1e-12), gamma)

    alpha = torch.where(~m, torch.atan2(R21, R11 + 1e-12), alpha)

    D_dict = {}
    for l in range(1, l_max + 1):
        D_dict[l] = wigner_D(l, alpha, beta, gamma)  # [N, 2l+1, 2l+1]
    return D_dict


# ------------------- 旋转类 -------------------
class IrrepsRotator:
    def __init__(self, irreps):
        self.irreps = o3.Irreps(irreps).simplify() if isinstance(irreps, str) else o3.Irreps(irreps).simplify()
        self.blocks = list(self.irreps)
        self.block_slices = []
        cur = 0
        for mul, ir in self.blocks:
            length = int(mul * ir.dim)
            self.block_slices.append((cur, length))
            cur += length
        self.total_dim = cur
        self.block_info = [(int(ir.l), int(ir.dim), int(mul)) for mul, ir in self.blocks]

        from collections import defaultdict
        l_to_cols, l_to_dim = defaultdict(list), {}
        for (start, length), (l, dim, mul) in zip(self.block_slices, self.block_info):
            l_to_dim[l] = dim
            if l == 0:
                l_to_cols[l].extend(range(start, start + length))
            else:
                for m_idx in range(mul):
                    s = start + m_idx * dim
                    e = s + dim
                    l_to_cols[l].extend(range(s, e))
        self._l_vals = sorted(l_to_cols.keys())
        self._l_idx_cpu = {l: torch.tensor(cols, dtype=torch.long) for l, cols in l_to_cols.items()}
        self._l_dim = l_to_dim
        self._l_mulsum = {l: (len(self._l_idx_cpu[l]) // self._l_dim[l]) if l != 0 else len(self._l_idx_cpu[l])
                          for l in self._l_vals}

    def rotate(self, D_dict, x):
        n = x.size(0)
        x_y = x.clone()
        for l in self._l_vals:
            if l == 0 or (l not in D_dict):
                continue
            idx = self._l_idx_cpu[l].to(x.device)
            d = self._l_dim[l]
            m = self._l_mulsum[l]
            if idx.numel() == 0:
                continue
            Xl = x.index_select(1, idx).view(n, m, d)
            Dl = D_dict[l].to(x.device)
            Xl_rot = torch.einsum('nji,nkj->nki', Dl, Xl)
            x_y[:, idx] = Xl_rot.reshape(n, m * d)
        return x_y

    def rotate_inv(self, D_dict, x_y):
        n = x_y.size(0)
        x_orig = x_y.clone()
        for l in self._l_vals:
            if l == 0 or (l not in D_dict):
                continue
            idx = self._l_idx_cpu[l].to(x_y.device)
            d = self._l_dim[l]
            m = self._l_mulsum[l]
            if idx.numel() == 0:
                continue
            Xl = x_y.index_select(1, idx).view(n, m, d)
            Dl = D_dict[l].to(x_y.device)
            Xl_inv = torch.einsum('nji,nki->nkj', Dl, Xl)
            x_orig[:, idx] = Xl_inv.reshape(n, m * d)
        return x_orig



if __name__ == "__main__":
    import torch
    from e3nn import o3

    # 假设之前定义好的 IrrepsRotator
    # rotator = IrrepsRotator(irreps)

    irreps = "2x0e + 1x1o + 2x2e"
    rotator = IrrepsRotator(irreps)

    n = 10
    # 随机生成 n 个向量
    vectors = torch.randn(n,3)

    # 生成对应 irreps 的 spherical harmonics
    # 注意 vectors[:, [1,2,0]] 做了顺序调整，使得 l=1 block 对应 [y,z,x] 约定
    x = o3.spherical_harmonics(
        irreps,
        vectors[:, [1,2,0]],
        normalize=True,
        normalization='component'
    )  # x: [n, total_dim]

    print("Before rotation (first 3 rows):")
    print(x[:3, :10])  # 打印前几个元素看看

    # 使用旋转函数，将 vectors 对应的 tensor 旋转到 z 方向
    x_z = rotator.rotate(vectors, x)

    print("\nAfter rotation to z (first 3 rows):")
    print(x_z[:3, :10])

    # 检查 l=1 block 是否对齐 z 方向
    for block_idx, (mul, ir) in enumerate(rotator.blocks):
        if int(ir.l)==1 and ir.p=='o':
            start, length = rotator.block_slices[block_idx]
            l1_block = x_z[:, start:start+3]
            print("\nl=1 block after rotation to z (should align with z-axis):")
            print(l1_block)
            xy = l1_block[:,:2]
            print("max abs of x/y components (should be ~0):", xy.abs().max().item())

    # 测试逆旋转
    x_back = rotator.rotate_inv(vectors, x_z)
    print("\nMax diff original <-> back:", (x - x_back).abs().max().item())

