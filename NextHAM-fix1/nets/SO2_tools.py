# -*- coding: utf-8 -*-
import torch
import math
from e3nn import o3
import torch.nn as nn

try:
    from .SO3_tools import IrrepsRotator
except ImportError:
    from SO3_tools import IrrepsRotator


# ===================== 工具：按 m 分组（保持你原样） =====================
class IrrepsMGroup_spherical:
    """
    针对球谐分量按 m 分组的工具（m 从 -lmax 到 +lmax）。
    transform:  x -> (x_0, x_m)
    inverse:   (x_0, x_m) -> xrec
    """

    def __init__(self, irreps):
        if isinstance(irreps, str):
            self.irreps = o3.Irreps(irreps).simplify()
        else:
            self.irreps = o3.Irreps(irreps).simplify()

        self.blocks = list(self.irreps)

        self.block_slices = []
        cur = 0
        for mul, ir in self.blocks:
            length = int(mul * ir.dim)
            self.block_slices.append((cur, length))
            cur += length
        self.total_dim = cur

        self.lmax = 0
        for mul, ir in self.blocks:
            if int(ir.l) > self.lmax:
                self.lmax = int(ir.l)

        self.ms = list(range(-self.lmax, self.lmax + 1))

        self.m_block_info = []
        for m in self.ms:
            entries = []
            for block_idx, (mul, ir) in enumerate(self.blocks):
                l = int(ir.l)
                dim = int(ir.dim)
                if abs(m) <= l:
                    base_start, base_len = self.block_slices[block_idx]
                    for copy_idx in range(int(mul)):
                        local_offset = copy_idx * dim
                        idx_in_copy = m + l
                        start = base_start + local_offset + idx_in_copy
                        entries.append((block_idx, copy_idx, start, 1))
            self.m_block_info.append(entries)

        self.m_dims = [len(entries) for entries in self.m_block_info]
        self.m_starts = [[start for (_b, _c, start, _d) in entries] for entries in self.m_block_info]
        self.m_to_index = {m: i for i, m in enumerate(self.ms)}

        self.x0_dim = 0
        if 0 in self.m_to_index:
            self.x0_dim = self.m_dims[self.m_to_index[0]]

        xm = {}
        for k in range(1, self.lmax + 1):
            dim_pos = self.m_dims[self.m_to_index[k]] if k in self.m_to_index else 0
            dim_neg = self.m_dims[self.m_to_index[-k]] if -k in self.m_to_index else 0
            xm[k] = max(dim_pos, dim_neg)
        self.xm_dims = xm

    def transform(self, x):
        assert x.dim() == 2, "x must be 2D [N, total_dim]"
        N = x.size(0)
        assert x.size(1) == self.total_dim, f"input dim {x.size(1)} != expected {self.total_dim}"

        device = x.device
        dtype = x.dtype

        parts = {}
        for i, m in enumerate(self.ms):
            starts = self.m_starts[i]
            if len(starts) == 0:
                parts[m] = torch.zeros((N, 0), dtype=dtype, device=device)
            else:
                idx = torch.tensor(starts, dtype=torch.long, device=device)
                parts[m] = x.index_select(1, idx)

        x_0 = parts.get(0, torch.zeros((N, 0), dtype=dtype, device=device))

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
            stacked = torch.stack([pos, neg], dim=1)
            x_m[k] = stacked

        return x_0, x_m

    def inverse(self, x_0, x_m):
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

        if 0 in self.m_to_index:
            starts0_list = self.m_starts[self.m_to_index[0]]
            if len(starts0_list) > 0 and x_0 is not None and x_0.numel() > 0:
                idx0 = torch.as_tensor(starts0_list, dtype=torch.long, device=device)
                assert x_0.dim() == 2 and x_0.size(1) == idx0.numel(), "x_0 dim mismatch"
                xrec[:, idx0] = x_0

        for k in range(1, self.lmax + 1):
            pair = x_m.get(k, None)
            if pair is None or pair.numel() == 0:
                continue
            assert pair.dim() == 3 and pair.size(1) == 2, f"x_m[{k}] must be [N,2,D]"

            pos = pair[:, 0, :]
            neg = pair[:, 1, :]

            if k in self.m_to_index:
                starts_pos = self.m_starts[self.m_to_index[k]]
                if len(starts_pos) > 0:
                    idx_pos = torch.as_tensor(starts_pos, dtype=torch.long, device=device)
                    Dp = idx_pos.numel()
                    if pos.size(1) >= Dp:
                        xrec[:, idx_pos] = pos[:, :Dp]
                    else:
                        if pos.size(1) > 0:
                            xrec[:, idx_pos[:pos.size(1)]] = pos

            if -k in self.m_to_index:
                starts_neg = self.m_starts[self.m_to_index[-k]]
                if len(starts_neg) > 0:
                    idx_neg = torch.as_tensor(starts_neg, dtype=torch.long, device=device)
                    Dn = idx_neg.numel()
                    if neg.size(1) >= Dn:
                        xrec[:, idx_neg] = neg[:, :Dn]
                    else:
                        if neg.size(1) > 0:
                            xrec[:, idx_neg[:neg.size(1)]] = neg

        return xrec


# ===================== 修改点 1：SO2_m_Convolution 支持置零 xr/xi =====================
class SO2_m_Convolution(nn.Module):
    """
    输入:  x_m [E, 2, n_in]
    输出:  [E, 2, n_out]

    zero_xi=True  -> 强制 x_i = 0（same-parity 分支）
    zero_xr=True  -> 强制 x_r = 0（opp-parity 分支）
    """
    def __init__(self, n_in, n_out, zero_xi=False, zero_xr=False):
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.zero_xi = bool(zero_xi)
        self.zero_xr = bool(zero_xr)
        assert not (self.zero_xi and self.zero_xr), "不能同时置零 x_i 和 x_r"

        self.fc = nn.Linear(self.n_in, 2 * self.n_out, bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m):
        assert x_m.dim() == 3 and x_m.size(1) == 2 and x_m.size(2) == self.n_in, \
            f"SO2_m_Convolution: 期望输入形状 [E,2,{self.n_in}]，得到 {tuple(x_m.shape)}"

        x_m = self.fc(x_m)  # [E, 2, 2*n_out]
        half = self.fc.out_features // 2
        x_r = x_m.narrow(2, 0, half)
        x_i = x_m.narrow(2, half, half)

        if self.zero_xi:
            x_i = torch.zeros_like(x_i)
        if self.zero_xr:
            x_r = torch.zeros_like(x_r)

        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)
        return torch.cat((x_m_r, x_m_i), dim=1)


# ===================== 修改点 2：SO2_Convolution 增加两个小参数（最小改动） =====================
class SO2_Convolution(nn.Module):
    """
    基本结构不变，只新增：
      - m_conv_mode: "same" -> x_i=0, "opp" -> x_r=0, "full" -> 不置零
      - allow_m0_out: False 时，m=0 输出强制为 0（用于 opp 网络）
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        fea_weight_dim,
        gate_dim=0,
        mlp_hidden=64,
        m_conv_mode="full",     # "same" / "opp" / "full"
        allow_m0_out=True       # opp 网络设为 False
    ):
        super().__init__()
        self.irreps_in  = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        self.rot_in  = IrrepsRotator(self.irreps_in)
        self.rot_out = IrrepsRotator(self.irreps_out)

        self.gm_in  = IrrepsMGroup_spherical(self.irreps_in)
        self.gm_out = IrrepsMGroup_spherical(self.irreps_out)
        self.lmax = max(self.gm_in.lmax, self.gm_out.lmax)

        self.gate_dim = int(gate_dim)
        self.allow_m0_out = bool(allow_m0_out)

        # --- m=0 分支（基本照抄你的）---
        in_x0  = self.gm_in.x0_dim
        out_x0 = self.gm_out.x0_dim

        if (out_x0 > 0 or self.gate_dim > 0) and in_x0 == 0:
            raise AssertionError("SO2_Convolution: 需要 m=0 输入通道 (irreps_in 必须含 l=0)")

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

        # --- m>0 分支（仅改：根据 m_conv_mode 选择置零方式）---
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

                zero_xi = (m_conv_mode == "same")
                zero_xr = (m_conv_mode == "opp")
                self.m_convs[str(k)] = SO2_m_Convolution(n_in, n_out, zero_xi=zero_xi, zero_xr=zero_xr)

    def forward(self, edge_fea, edge_vec, D_dict, fea_weight):
        E = edge_fea.size(0)

        # 1) 旋到 z 框
        x_z = self.rot_in.rotate(D_dict, edge_fea)

        # 2) 按 m 分组
        x0_in, xm_in = self.gm_in.transform(x_z)

        # --- m=0 ---
        if self.gm_out.x0_dim > 0 or self.gate_dim > 0:
            w0 = self.fc_weight(fea_weight)
            x0_gated = x0_in * w0

            if self.gm_out.x0_dim > 0 and self.allow_m0_out:
                x0_out = self.lin_x0_out(x0_gated)
            else:
                # allow_m0_out=False 时，强制 m=0 输出为 0（满足你“m=0 非线性只用于 same”）
                x0_out = torch.zeros(E, self.gm_out.x0_dim, device=x0_in.device, dtype=x0_in.dtype)

            if self.gate_dim > 0:
                gate_in = self.lin_x0_gate(x0_gated)
            else:
                gate_in = torch.zeros(E, 0, device=x0_in.device, dtype=x0_in.dtype)
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
            xm_out[k] = conv(xk)

        # 3) 组回按 l 排（仍在 z 框）
        x_out_z = self.gm_out.inverse(x0_out, xm_out)

        # 4) 旋回原坐标
        edge_out = self.rot_out.rotate_inv(D_dict, x_out_z)

        return edge_out, gate_in


# ===================== 新增：wrapper，把输出按 parity 拆两部分再合并 =====================
def _is_same_as_sph_harm(l: int, ir_parity_int: int) -> bool:
    """
    球谐 Y_l 在反演下: (-1)^l
    ir_parity_int: e=+1, o=-1
    same <=> ir_parity_int == (-1)^l
    """
    sph = 1 if (l % 2 == 0) else -1
    return (ir_parity_int == sph)


class SO2_Convolution_ParitySplit(nn.Module):
    """
    外部用法与单个 SO2_Convolution 一样，但内部：
      - same_net 输出 natural parity (0e,1o,2e,...), m>0 强制 x_i=0, 且允许 m=0 非线性输出
      - opp_net  输出 unnatural parity (0o,1e,2o,...), m>0 强制 x_r=0, 且 m=0 输出强制为 0
    最后把两部分按 (simplify 后) irreps_out 的块顺序拼成完整输出向量。
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
        fea_weight_dim,
        gate_dim=0,
        mlp_hidden=64
    ):
        super().__init__()

        self.irreps_in  = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        # 把输出 irreps 按 parity 拆成两份（同样 simplify 后）
        same_blocks = []
        opp_blocks  = []
        for mul, ir in list(self.irreps_out):
            l = int(ir.l)
            p = int(ir.p)  # e=+1, o=-1
            if _is_same_as_sph_harm(l, p):
                same_blocks.append((mul, ir))
            else:
                opp_blocks.append((mul, ir))

        self.irreps_out_same = o3.Irreps(same_blocks) if len(same_blocks) > 0 else o3.Irreps([])
        self.irreps_out_opp  = o3.Irreps(opp_blocks)  if len(opp_blocks)  > 0 else o3.Irreps([])

        # 两个子网络
        self.same_net = SO2_Convolution(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out_same,
            fea_weight_dim=fea_weight_dim,
            gate_dim=gate_dim,          # gate 只从 same_net 出
            mlp_hidden=mlp_hidden,
            m_conv_mode="same",         # x_i = 0
            allow_m0_out=True           # m=0 非线性只在 same
        )

        self.opp_net = SO2_Convolution(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out_opp,
            fea_weight_dim=fea_weight_dim,
            gate_dim=0,                 # opp 不输出 gate（你也可以改成需要的话再加）
            mlp_hidden=mlp_hidden,
            m_conv_mode="opp",          # x_r = 0
            allow_m0_out=False          # m=0 输出强制为 0
        )

        # 预计算：把 same/opp 子输出拼回 full 输出的 slice 映射（按 simplify 后 block 顺序）
        self._full_slices = []
        cur = 0
        for mul, ir in list(self.irreps_out):
            d = int(mul * ir.dim)
            self._full_slices.append((cur, cur + d, mul, ir))
            cur += d
        self.full_dim = cur

        # same 输出的块顺序也是 simplify 后的 same_irreps blocks
        self._same_slices = []
        cur = 0
        for mul, ir in list(self.irreps_out_same):
            d = int(mul * ir.dim)
            self._same_slices.append((cur, cur + d, mul, ir))
            cur += d
        self.same_dim = cur

        self._opp_slices = []
        cur = 0
        for mul, ir in list(self.irreps_out_opp):
            d = int(mul * ir.dim)
            self._opp_slices.append((cur, cur + d, mul, ir))
            cur += d
        self.opp_dim = cur

        # 为 full 的每个 block 标记来自 same 还是 opp，并给出在子输出里的 slice
        # 由于 simplify 可能合并 multiplicity，我们用 (l,p,mul) 的顺序匹配即可（同一列表内顺序一致）
        same_ptr = 0
        opp_ptr = 0
        self._assign = []  # list of ("same"/"opp", full_slice, sub_slice)
        for (fs, fe, fmul, fir) in self._full_slices:
            l = int(fir.l); p = int(fir.p)
            if _is_same_as_sph_harm(l, p):
                ss, se, smul, sir = self._same_slices[same_ptr]
                same_ptr += 1
                # 理论上 sir 应该等于 fir
                self._assign.append(("same", (fs, fe), (ss, se)))
            else:
                os, oe, omul, oir = self._opp_slices[opp_ptr]
                opp_ptr += 1
                self._assign.append(("opp", (fs, fe), (os, oe)))

    def forward(self, edge_fea, edge_vec, D_dict, fea_weight):
        # 两个子网络分别算各自输出（m>0 计算量不会翻倍，因为各自的 irreps_out 维度拆小了）
        out_same, gate_in = self.same_net(edge_fea, edge_vec, D_dict, fea_weight)
        out_opp, _ = self.opp_net(edge_fea, edge_vec, D_dict, fea_weight)

        E = edge_fea.size(0)
        device = edge_fea.device
        dtype = edge_fea.dtype
        out_full = torch.zeros((E, self.full_dim), device=device, dtype=dtype)

        # 拼回 full 输出
        for tag, (fs, fe), (ss, se) in self._assign:
            if tag == "same":
                if out_same.numel() > 0:
                    out_full[:, fs:fe] = out_same[:, ss:se]
            else:
                if out_opp.numel() > 0:
                    out_full[:, fs:fe] = out_opp[:, ss:se]

        return out_full, gate_in


if __name__ == "__main__":
    import torch
    from e3nn import o3
    from SO3_tools import compute_D

    torch.set_default_dtype(torch.float64)  # 建议用 double 更敏感
    torch.manual_seed(0)

    irreps_in  = "64x0e + 32x1o + 16x2e"
    irreps_out = "32x0e + 16x0o + 16x1o + 16x1e + 8x2e + 8x2o"

    E = 12
    fea_weight_dim = 64
    gate_dim = 32

    model = SO2_Convolution_ParitySplit(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        fea_weight_dim=fea_weight_dim,
        gate_dim=gate_dim,
        mlp_hidden=64,
    ).eval()

    # ------- 原始输入 -------
    edge_vec   = torch.randn(E, 3)
    edge_fea   = torch.randn(E, o3.Irreps(irreps_in).simplify().dim)
    fea_weight = torch.randn(E, fea_weight_dim)  # 标量特征：O3 下不变

    # 对应 D_dict
    lmax_for_D = 7
    D_dict = compute_D(edge_vec, l_max=lmax_for_D)

    with torch.no_grad():
        y, gate = model(edge_fea, edge_vec, D_dict, fea_weight)

    # ==========================================================
    # 生成一个随机 O(3) 变换 g = (R, sigma)
    # ==========================================================
    R = o3.rand_matrix()  # [3,3] 随机旋转
    sigma = -1.0          # 改成 +1.0 测 SO(3)；-1.0 测含反演的 O(3)

    R = R * sigma

    # 变换 edge_vec：行向量形式 v -> sigma * v R^T
    edge_vec_g = (edge_vec[:, [1,2,0]] @ o3.Irreps("1x1o").D_from_matrix(R))[:,[2,0,1]]

    # 变换 edge_fea：用 irreps_in 的表示矩阵 D_in(g)
    # D_from_matrix 支持 parity 参数 p：p=-1 表示额外反演
    irreps_in_s = o3.Irreps(irreps_in).simplify()
    irreps_out_s = o3.Irreps(irreps_out).simplify()

    D_in  = irreps_in_s.D_from_matrix(R)   # [din, din]
    D_out = irreps_out_s.D_from_matrix(R)  # [dout, dout]

    # 特征是 row-major: x -> x @ D^T
    edge_fea_g = edge_fea @ D_in

    # 新的 D_dict（因为输入向量变了）
    D_dict_g = compute_D(edge_vec_g, l_max=lmax_for_D)

    with torch.no_grad():
        y_g, gate_g = model(edge_fea_g, edge_vec_g, D_dict_g, fea_weight)

    # 期望输出：y_expected = (g ⋅ y) = y @ D_out^T
    y_expected = y @ D_out

    # ==========================================================
    # 误差评估
    # ==========================================================
    abs_err = (y_g - y_expected).abs()
    max_abs = abs_err.max().item()
    rms_abs = abs_err.pow(2).mean().sqrt().item()

    denom = y_expected.abs().max().item() + 1e-12
    rel_max = max_abs / denom
    rel_rms = rms_abs / (y_expected.pow(2).mean().sqrt().item() + 1e-12)

    print(f"[O(3) equiv test] sigma={int(sigma)}")
    print("y shape:", y.shape)
    print("max_abs_err:", max_abs)
    print("rms_abs_err:", rms_abs)
    print("rel_max_err:", rel_max)
    print("rel_rms_err:", rel_rms)

    # gate 的等变性：
    # gate 通常设计成标量/不变的（0e），如果你希望检验它不变：
    # 这里简单检查 gate_g 与 gate 是否一致（因为它只来自 m=0 标量分支）
    if gate.numel() > 0:
        gate_abs = (gate_g - gate).abs().max().item()
        gate_rel = gate_abs / (gate.abs().max().item() + 1e-12)
        print("gate max_abs_err:", gate_abs)
        print("gate rel_err:", gate_rel)

