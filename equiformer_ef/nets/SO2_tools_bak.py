# -*- coding: utf-8 -*-
import torch
import math
from e3nn import o3
from e3nn.o3 import TensorProduct, Linear
from e3nn.nn import Gate
import torch.nn as nn
try:
    from .SO3_tools import IrrepsRotator
except ImportError:
    from SO3_tools import IrrepsRotator


import time

def _now_and_sync(tensor_like: torch.Tensor):
    if tensor_like.is_cuda:
        torch.cuda.synchronize(tensor_like.device)
    return time.perf_counter()

class IrrepsMGroup_spherical:
    """
    针对球谐分量按 m 分组的工具（m 从 -lmax 到 +lmax）。
    主要接口：
      gm = IrrepsMGroup_spherical("2x0e + 1x1e + 2x2e")
      print(gm.x0_dim)        # m=0 的通道数
      print(gm.xm_dims)       # dict: k -> D_k（x_m[k] 的第二/第三维度）
      x_0, x_m = gm.transform(x)
      xrec = gm.inverse(x_0, x_m)
    """

    def __init__(self, irreps):
        if isinstance(irreps, str):
            self.irreps = o3.Irreps(irreps).simplify()
        else:
            self.irreps = o3.Irreps(irreps).simplify()

        # 原 blocks：列表 (mul, Irrep)
        self.blocks = list(self.irreps)

        # 每个 block 在原向量上的起点与长度（按原顺序）
        self.block_slices = []
        cur = 0
        for mul, ir in self.blocks:
            length = int(mul * ir.dim)  # mul * (2l+1)
            self.block_slices.append((cur, length))
            cur += length
        self.total_dim = cur

        # l_max（所有 blocks 中的最大 l）
        self.lmax = 0
        for mul, ir in self.blocks:
            if int(ir.l) > self.lmax:
                self.lmax = int(ir.l)

        # 全部 m 值从 -lmax 到 +lmax
        self.ms = list(range(-self.lmax, self.lmax + 1))

        # 构建 m -> entries 映射
        # m_block_info: per m: list of (block_idx, copy_idx, start, 1)
        self.m_block_info = []
        for m in self.ms:
            entries = []
            for block_idx, (mul, ir) in enumerate(self.blocks):
                l = int(ir.l)
                dim = int(ir.dim)  # equals 2*l + 1
                if abs(m) <= l:
                    base_start, base_len = self.block_slices[block_idx]
                    for copy_idx in range(int(mul)):
                        local_offset = copy_idx * dim
                        idx_in_copy = m + l
                        start = base_start + local_offset + idx_in_copy
                        entries.append((block_idx, copy_idx, start, 1))
            self.m_block_info.append(entries)

        # 每个 m 的维度（number of entries）
        self.m_dims = [len(entries) for entries in self.m_block_info]

        # 每个 m 对应的列索引 starts（方便一次性 index_select）
        self.m_starts = [[start for (_b, _c, start, _d) in entries] for entries in self.m_block_info]

        # m -> index 映射（便于查询）
        self.m_to_index = {m: i for i, m in enumerate(self.ms)}

        # 预计算 x0_dim 与 xm_dims（在 transform 前可读）
        self.x0_dim = 0
        if 0 in self.m_to_index:
            self.x0_dim = self.m_dims[self.m_to_index[0]]
        # xm_dims: dict k->D_k, D_k = max(dim_{+k}, dim_{-k})
        xm = {}
        for k in range(1, self.lmax + 1):
            dim_pos = self.m_dims[self.m_to_index[k]] if k in self.m_to_index else 0
            dim_neg = self.m_dims[self.m_to_index[-k]] if -k in self.m_to_index else 0
            xm[k] = max(dim_pos, dim_neg)
        self.xm_dims = xm  # dict: k -> D_k

    def shapes(self):
        # 兼容旧接口：返回每 m 的维度（list）
        return list(self.m_dims)

    def transform(self, x):
        """
        直接返回 x_0, x_m
          - x_0: [N, x0_dim]
          - x_m: dict: k -> [N, 2, D_k]  (idx0 = +k, idx1 = -k)
        """
        assert x.dim() == 2, "x must be 2D [N, total_dim]"
        N = x.size(0)
        assert x.size(1) == self.total_dim, f"input dim {x.size(1)} != expected {self.total_dim}"

        device = x.device
        dtype = x.dtype

        # 先一次性按 m_starts 做 index_select 构建 parts map：m -> [N, dim_m]
        parts = {}
        for i, m in enumerate(self.ms):
            starts = self.m_starts[i]
            if len(starts) == 0:
                parts[m] = torch.zeros((N, 0), dtype=dtype, device=device)
            else:
                idx = torch.tensor(starts, dtype=torch.long, device=device)
                parts[m] = x.index_select(1, idx)  # [N, dim_m]

        # x_0
        x_0 = parts.get(0, torch.zeros((N, 0), dtype=dtype, device=device))

        # 构造 x_m: 每个 k -> stack(+k, -k) with padding to D_k
        x_m = {}
        for k in range(1, self.lmax + 1):
            pos = parts.get(k, torch.zeros((N, 0), dtype=dtype, device=device))
            neg = parts.get(-k, torch.zeros((N, 0), dtype=dtype, device=device))
            D_pos = pos.size(1)
            D_neg = neg.size(1)
            D = self.xm_dims[k]
            if D_pos < D:
                pad = torch.zeros((N, D - D_pos), dtype=dtype, device=device)
                pos = torch.cat([pos, pad], dim=1)
            if D_neg < D:
                pad = torch.zeros((N, D - D_neg), dtype=dtype, device=device)
                neg = torch.cat([neg, pad], dim=1)
            stacked = torch.stack([pos, neg], dim=1)  # [N,2,D]
            x_m[k] = stacked

        return x_0, x_m

    def inverse(self, x_0, x_m):
        # ------- 推断 N/device/dtype -------
        N = 0
        device = None
        dtype = None
        if x_0 is not None and x_0.numel() > 0:
            N = x_0.size(0); device = x_0.device; dtype = x_0.dtype
        else:
            for v in x_m.values():
                if v is not None and v.numel() > 0:
                    N = v.size(0); device = v.device; dtype = v.dtype; break
            if device is None:
                device = torch.device("cpu"); dtype = torch.float32

        xrec = torch.zeros((N, self.total_dim), dtype=dtype, device=device)

        # ------- m == 0：整块写回（无内层循环） -------
        if 0 in self.m_to_index:
            starts0_list = self.m_starts[self.m_to_index[0]]  # Python list of ints
            if len(starts0_list) > 0 and x_0 is not None and x_0.numel() > 0:
                idx0 = torch.as_tensor(starts0_list, dtype=torch.long, device=device)  # [D0]
                assert x_0.dim() == 2 and x_0.size(1) == idx0.numel(), "x_0 dim mismatch"
                # 一次性写回所有列
                xrec[:, idx0] = x_0

        # ------- k >= 1：对 +k / -k 分别整块写回 -------
        for k in range(1, self.lmax + 1):
            pair = x_m.get(k, None)
            if pair is None or pair.numel() == 0:
                continue
            assert pair.dim() == 3 and pair.size(1) == 2, f"x_m[{k}] must be [N,2,D]"

            pos = pair[:, 0, :]  # [N, D]
            neg = pair[:, 1, :]  # [N, D]

            # +k
            if k in self.m_to_index:
                starts_pos = self.m_starts[self.m_to_index[k]]  # list[int]
                if len(starts_pos) > 0:
                    idx_pos = torch.as_tensor(starts_pos, dtype=torch.long, device=device)  # [Dp]
                    Dp = idx_pos.numel()
                    if pos.size(1) >= Dp:
                        xrec[:, idx_pos] = pos[:, :Dp]
                    else:
                        # pos 比 idx_pos 短：只写已有的列（其余保持为 0）
                        if pos.size(1) > 0:
                            xrec[:, idx_pos[:pos.size(1)]] = pos

            # -k
            if -k in self.m_to_index:
                starts_neg = self.m_starts[self.m_to_index[-k]]
                if len(starts_neg) > 0:
                    idx_neg = torch.as_tensor(starts_neg, dtype=torch.long, device=device)  # [Dn]
                    Dn = idx_neg.numel()
                    if neg.size(1) >= Dn:
                        xrec[:, idx_neg] = neg[:, :Dn]
                    else:
                        if neg.size(1) > 0:
                            xrec[:, idx_neg[:neg.size(1)]] = neg

        return xrec

class SO2_m_Convolution(nn.Module):
    """
    输入:  x_m [E, 2, n_in]   # 2 表示 (+m, -m) 的“复数对”
    输出:  [E, 2, n_out]
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.fc = nn.Linear(self.n_in, 2 * self.n_out, bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m):
        # x_m: [E, 2, n_in]
        assert x_m.dim() == 3 and x_m.size(1) == 2 and x_m.size(2) == self.n_in, \
            f"SO2_m_Convolution: 期望输入形状 [E,2,{self.n_in}]，得到 {tuple(x_m.shape)}"
        x_m = self.fc(x_m)  # [E, 2, 2*n_out]
        half = self.fc.out_features // 2
        x_r = x_m.narrow(2, 0, half)                # [E, 2, n_out]
        x_i = x_m.narrow(2, half, half)             # [E, 2, n_out]
        # 复数线性（保持 SO(2) 结构）
        # x_i = torch.zeros_like(x_i)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)   # [E,1,n_out]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)   # [E,1,n_out]
        return torch.cat((x_m_r, x_m_i), dim=1)             # [E,2,n_out]


# --------------------------- SO2_Convolution 主结构 ---------------------------
class SO2_Convolution(nn.Module):
    """
      edge_fea --(rotate到z)---> x_z
      x_z --(IrrepsMGroup_spherical.transform)--> (x0_in, xm_in)
        m=0: fea_weight -> MLP -> 权重；(x0_in * w0) --lin_x0_out--> out 的 m=0
                                   (x0_in * w0) --lin_x0_gate--> gate_input
        m>0: 逐 k 用 SO2_m_Convolution，从 D_in[k] -> D_out[k]
      (x0_out, xm_out) --inverse(m-group)--> x_out_z
      x_out_z --(rotate_inv 旋回原坐标)--> edge_out
    返回: (edge_out, gate_input)
    """
    def __init__(
        self,
        irreps_in,              # str / o3.Irreps
        irreps_out,             # str / o3.Irreps
        fea_weight_dim,         # fea_weight 的输入维度
        gate_dim=0,             # gate 输出维度（来自 m=0 分支的独立头）
        mlp_hidden=64           # 生成 m=0 权重的隐藏层
    ):
        super().__init__()
        self.irreps_in  = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()

        self.rot_in  = IrrepsRotator(self.irreps_in)
        self.rot_out = IrrepsRotator(self.irreps_out)

        # m-分组
        self.gm_in  = IrrepsMGroup_spherical(self.irreps_in)
        self.gm_out = IrrepsMGroup_spherical(self.irreps_out)
        self.lmax = max(self.gm_in.lmax, self.gm_out.lmax)

        # --- m=0 分支 ---
        self.gate_dim = int(gate_dim)
        in_x0  = self.gm_in.x0_dim
        out_x0 = self.gm_out.x0_dim

        if out_x0 > 0 or self.gate_dim > 0:
            assert in_x0 > 0, "SO2_Convolution: 需要 m=0 输入通道 (irreps_in 必须含 l=0)"

        if in_x0 > 0:
            self.fc_weight = nn.Sequential(
                nn.Linear(fea_weight_dim, mlp_hidden),
                torch.nn.LayerNorm(mlp_hidden, eps=1e-6),
                nn.SiLU(),
                nn.Linear(mlp_hidden, in_x0),
            )
            self.lin_x0_out  = nn.Linear(in_x0, out_x0, bias=True) if out_x0 > 0 else nn.Identity()
            self.lin_x0_gate = nn.Linear(in_x0, self.gate_dim, bias=True) if self.gate_dim > 0 else nn.Identity()
        else:
            self.fc_weight = nn.Identity()
            self.lin_x0_out  = nn.Identity()
            self.lin_x0_gate = nn.Identity()

        # --- m>0 分支：仅为 in/out 都存在的 k 构建算子（否则直接 assert）---
        self.m_convs = nn.ModuleDict()
        for k in range(1, self.lmax + 1):
            n_in  = self.gm_in.xm_dims.get(k, 0)
            n_out = self.gm_out.xm_dims.get(k, 0)
            if n_out == 0 and n_in == 0:
                continue  
            if n_out == 0 and n_in > 0:
                continue
            if n_out > 0:
                assert n_in > 0, f"SO2_Convolution: 需要 k={k} 的输入通道 (irreps_in 必须含 l>= {k})"
                self.m_convs[str(k)] = SO2_m_Convolution(n_in, n_out)

   
    def forward(self, edge_fea, edge_vec, D_dict, fea_weight):
        """
        edge_fea:   [E, dim(irreps_in)]  （按-l..l 排列）
        edge_vec:   [E, 3]
        fea_weight: [E, fea_weight_dim]
        返回:
          edge_out:   [E, dim(irreps_out)]  （按-l..l 排列，且已旋回原坐标）
          gate_input: [E, gate_dim]
        """
        E = edge_fea.size(0)
        # edge_fea[:, [2,3,4]] = edge_vec[:,[1,2,0]]

        # 1) 旋到 z 框
        x_z = self.rot_in.rotate(D_dict, edge_fea)                    # [E, dim_in]

        # 2) 按 m 分组
        # import pdb;pdb.set_trace()
        x0_in, xm_in = self.gm_in.transform(x_z)
        # x0_in: [E, in_x0]

        # --- m=0 ---
        if self.gm_out.x0_dim > 0 or self.gate_dim > 0:
            assert x0_in.size(1) == self.gm_in.x0_dim, "m=0 输入通道数不匹配"
            w0 = self.fc_weight(fea_weight)                                # [E, in_x0]
            assert w0.size(1) == x0_in.size(1), "m=0 权重与输入通道不匹配"
            x0_gated = x0_in * w0                                          # [E, in_x0]
            x0_out   = self.lin_x0_out(x0_gated) if self.gm_out.x0_dim > 0 else torch.zeros(E, 0, device=x0_in.device, dtype=x0_in.dtype)
            gate_in  = self.lin_x0_gate(x0_gated) if self.gate_dim > 0     else torch.zeros(E, 0, device=x0_in.device, dtype=x0_in.dtype)
        else:
            x0_out  = torch.zeros(E, 0, device=edge_fea.device, dtype=edge_fea.dtype)
            gate_in = torch.zeros(E, 0, device=edge_fea.device, dtype=edge_fea.dtype)

        # --- m>0 ---
        xm_out = {}
        for k, conv in self.m_convs.items():
            k = int(k)
            xk = xm_in.get(k, None)
            assert xk is not None, f"缺少 k={k} 的输入分量"
            n_in = conv.n_in
            assert xk.dim() == 3 and xk.size(1) == 2 and xk.size(2) == n_in, \
                f"k={k} 输入形状应为 [E,2,{n_in}]，实际 {tuple(xk.shape)}"
            yk = conv(xk)                                                  # [E,2,n_out]
            xm_out[k] = yk

        # 3) 组回按 l 排（仍在 z 框）
        # x0_out = torch.zeros_like(x0_out)
        x_out_z = self.gm_out.inverse(x0_out, xm_out)
        # import pdb;pdb.set_trace()                      # [E, dim_out]

        # 4) 旋回原坐标
        edge_out = self.rot_out.rotate_inv(D_dict, x_out_z)              # [E, dim_out]

        return edge_out, gate_in
    
    

if __name__ == "__main__":
    import torch
    from e3nn import o3

    irreps_in  = "64x0e + 32x1o + 16x2e"
    irreps_out = "64x0e + 32x1o + 16x2e"

    E = 12
    fea_weight_dim = 64
    gate_dim = 32

    model = SO2_Convolution(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        fea_weight_dim=fea_weight_dim,
        gate_dim=gate_dim,
        mlp_hidden=64,
    )

    edge_vec   = torch.randn(E, 3)
    edge_fea   = torch.randn(E, o3.Irreps(irreps_in).dim)
    fea_weight = torch.randn(E, fea_weight_dim)

    edge_out, gate_in = model(edge_fea, edge_vec, fea_weight)
    print("edge_out:", edge_out.shape)   # -> [E, dim(irreps_out)]
    print("gate_in:",  gate_in.shape)    # -> [E, gate_dim]

