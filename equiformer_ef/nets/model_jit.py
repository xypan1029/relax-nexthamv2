import warnings
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch_scatter import scatter  # Python/JIT 可用；C++ 侧需携带相应 .so/.dll
from e3nn import o3
from e3nn.o3 import TensorProduct, Linear
from e3nn.nn import Gate
from e3nn.util.jit import compile_mode, script

# ---------- 实用函数 ----------
def _parse_irreps(irreps: o3.Irreps) -> Tuple[List[int], List[int], int]:
    muls: List[int] = []
    dims: List[int] = []
    for mul, ir in irreps:
        muls.append(int(mul))
        dims.append(int(ir.dim))
    total = int(sum(m * d for m, d in zip(muls, dims)))
    return muls, dims, total

@torch.jit.ignore
def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    """仅构造期使用；不进入 JIT 图。"""
    irr1 = o3.Irreps(irreps_in1).simplify()
    irr2 = o3.Irreps(irreps_in2).simplify()
    ir_o = o3.Irrep(ir_out)
    for _, ir1 in irr1:
        for _, ir2 in irr2:
            if ir_o in (ir1 * ir2):
                return True
    return False

# ---------- Gate 工厂（JIT 友好，不暴露函数字典） ----------
def _sel_act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "silu":     return torch.nn.functional.silu
    if name == "tanh":     return torch.tanh
    if name == "sigmoid":  return torch.sigmoid
    if name == "identity": return lambda x: x
    raise ValueError(f"Unknown activation: {name}")

def get_gate_stage2_jit(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    act_scalar_pos: str = "silu",
    act_scalar_neg: str = "tanh",
    act_gate_pos: str   = "sigmoid",
    act_gate_neg: str   = "tanh",
) -> Gate:
    irreps_out = o3.Irreps(irreps_out)

    irreps_scalars = o3.Irreps(
        (mul, ir) for mul, ir in irreps_out
        if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ).simplify()

    irreps_gated = o3.Irreps(
        (mul, ir) for mul, ir in irreps_out
        if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ).simplify()

    if irreps_gated.dim > 0:
        if tp_path_exists(irreps_in1, irreps_in2, "0e"):
            gate_ir = "0e"
        elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
            gate_ir = "0o"; warnings.warn("Using odd representations as gates")
        else:
            raise ValueError("No scalar gate path for requested irreps_gated")
    else:
        gate_ir = None

    irreps_gates = o3.Irreps([(mul, gate_ir) for mul, _ in irreps_gated]).simplify()

    def pick_scalar(p: int): return _sel_act(act_scalar_pos if p == 1 else act_scalar_neg)
    def pick_gate(p: int):   return _sel_act(act_gate_pos   if p == 1 else act_gate_neg)

    act_scalars = [pick_scalar(ir.p) for _, ir in irreps_scalars]
    act_gates   = [pick_gate(ir.p)   for _, ir in irreps_gates]

    return Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
# 参考：Gate 的 API 接收 act 列表，e3nn 的 JIT 封装会处理混编。:contentReference[oaicite:4]{index=4}

# ---------- E3ElementWise：逐字段缩放（trace） ----------
@compile_mode('script')
class E3ElementWise(nn.Module):
    muls: List[int]; dims: List[int]
    len_weight: int; out_dim: int

    def __init__(self, irreps_in: str):
        super().__init__()
        ir = o3.Irreps(irreps_in)
        muls, dims, total = _parse_irreps(ir)
        self.muls = muls; self.dims = dims
        self.len_weight = int(sum(muls))
        self.out_dim = total

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        N = x.size(0); ix = 0; iw = 0
        outs: List[torch.Tensor] = []
        for i in range(len(self.muls)):
            mul_i, dim_i = self.muls[i], self.dims[i]
            span = mul_i * dim_i
            field = x[:, ix: ix+span].reshape(N, mul_i, dim_i)
            ww = w[:, iw: iw+mul_i].unsqueeze(-1)   # [N, mul, 1]
            outs.append((field * ww).reshape(N, span))
            ix += span; iw += mul_i
        return torch.cat(outs, dim=-1)

# ---------- 外部权重的 TensorProduct（trace） ----------
@compile_mode('trace')
class SeparateWeightTensorProduct(nn.Module):
    in1_muls: List[int]; in1_dims: List[int]; in1_dim: int
    in2_muls: List[int]; in2_dims: List[int]; in2_dim: int
    out_muls: List[int]; out_dims: List[int]; out_dim: int

    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        super().__init__()
        kwargs.pop('internal_weights', None)
        kwargs.pop('shared_weights',   None)

        irr1 = o3.Irreps(irreps_in1); irr2 = o3.Irreps(irreps_in2); irr3 = o3.Irreps(irreps_out)
        self.in1_muls, self.in1_dims, self.in1_dim = _parse_irreps(irr1)
        self.in2_muls, self.in2_dims, self.in2_dim = _parse_irreps(irr2)
        self.out_muls, self.out_dims, self.out_dim = _parse_irreps(irr3)

        instr_tp = []; w1s = []; w2s = []
        for i1, (m1, ir1) in enumerate(irr1):
            for i2, (m2, ir2) in enumerate(irr2):
                for io, (mo, ir3_) in enumerate(irr3):
                    if ir3_ in (ir1 * ir2):
                        w1s.append(nn.Parameter(torch.randn(m1, mo)))
                        w2s.append(nn.Parameter(torch.randn(m2, mo)))
                        instr_tp.append((i1, i2, io, 'uvw', True, 1.0))

        self.tp = TensorProduct(irr1, irr2, irr3, instr_tp,
                                internal_weights=False, shared_weights=True, **kwargs)
        self.weights1 = nn.ParameterList(w1s)
        self.weights2 = nn.ParameterList(w2s)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        flats: List[torch.Tensor] = []
        for w1, w2 in zip(self.weights1, self.weights2):
            flats.append((w1[:, None, :] * w2[None, :, :]).reshape(-1))
        weights = torch.cat(flats)
        return self.tp(x1, x2, weights)

    # 便于 e3nn 自动 tracing
    def _make_tracing_inputs(self, n: int):
        inputs = []
        N = 8
        for _ in range(n):
            x1 = torch.randn(N, self.in1_dim)
            x2 = torch.randn(N, self.in2_dim)
            inputs.append({'forward': (x1, x2)})
        return inputs

# ---------- e3LayerNorm（script） ----------
@compile_mode('script')
class e3LayerNorm(nn.Module):
    muls: List[int]; dims: List[int]; total_dim: int
    eps: float; subtract_mean: bool; divide_norm: bool
    use_norm_mode_norm: bool
    w_starts: List[int]; b_starts: List[int]

    def __init__(self, irreps_in, eps=1e-5, affine=True,
                 normalization='component', subtract_mean=True, divide_norm=False):
        super().__init__()
        irr = o3.Irreps(irreps_in).simplify()
        muls, dims, total = _parse_irreps(irr)
        self.muls, self.dims, self.total_dim = muls, dims, total
        self.eps = float(eps)
        self.subtract_mean = bool(subtract_mean)
        self.divide_norm = bool(divide_norm)
        if normalization not in ('component', 'norm'):
            raise ValueError("normalization must be 'component' or 'norm'")
        self.use_norm_mode_norm = (normalization == 'norm')

        w_starts: List[int] = []; b_starts: List[int] = []
        ix_w = 0; ix_b = 0
        for mul, ir in irr:
            w_starts.append(ix_w); ix_w += int(mul)
            if ir.is_scalar():
                b_starts.append(ix_b); ix_b += int(mul)
            else:
                b_starts.append(-1)
        self.w_starts = w_starts; self.b_starts = b_starts

        if affine:
            self.weight = nn.Parameter(torch.ones(ix_w))
            self.bias   = nn.Parameter(torch.zeros(ix_b)) if ix_b > 0 else nn.Parameter(torch.empty(0))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias',   None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None: self.weight.data.fill_(1)
        if self.bias   is not None and self.bias.numel() > 0: self.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = x.size(0)
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=x.device)
        B = int(batch.max().item()) + 1

        deg = scatter(torch.ones(N, dtype=x.dtype, device=x.device),
                      batch, dim=0, dim_size=B, reduce='sum')  # [B]

        outs: List[torch.Tensor] = []
        ix = 0
        for i in range(len(self.muls)):
            mul_i, dim_i = self.muls[i], self.dims[i]
            span = mul_i * dim_i
            field = x[:, ix: ix+span].reshape(N, mul_i, dim_i)

            if self.subtract_mean or dim_i == 1:
                s = scatter(field, batch, dim=0, dim_size=B, reduce='sum')
                mean = s / (deg.view(-1, 1, 1) + 1e-12)
                field = field - mean[batch]

            if self.divide_norm or dim_i == 1:
                sq = (field.abs().pow(2))
                m = scatter(sq, batch, dim=0, dim_size=B, reduce='mean')
                norm = m.mean(dim=[1, 2], keepdim=True)
                if self.use_norm_mode_norm: norm = norm * dim_i
                field = field / (norm.sqrt()[batch] + self.eps)

            if self.weight is not None:
                ws = self.w_starts[i]
                w = self.weight[ws: ws+mul_i].view(1, mul_i, 1)
                field = field * w
            if self.bias is not None:
                bs = self.b_starts[i]
                if bs >= 0:
                    b = self.bias[bs: bs+mul_i].view(1, mul_i, 1)
                    field = field + b

            outs.append(field.reshape(N, span)); ix += span

        return torch.cat(outs, dim=-1)

# ---------- 顶层 EquiConv（script） ----------
@compile_mode('script')
class EquiConv(nn.Module):
    use_norm: bool; nonlin_on: bool

    def __init__(self, fc_len_in, irreps_in1, irreps_in2, irreps_out,
                 norm: str = '', nonlin: bool = True):
        super().__init__()
        self.use_norm = bool(norm != '')
        self.nonlin_on = bool(nonlin)

        irr1 = o3.Irreps(irreps_in1); irr2 = o3.Irreps(irreps_in2); irr_out = o3.Irreps(irreps_out)

        if self.nonlin_on:
            self.nonlin = get_gate_stage2_jit(irr1, irr2, irr_out)
            irr_tp_out = self.nonlin.irreps_in
        else:
            irr_tp_out = o3.Irreps([(m, ir) for m, ir in irr_out if tp_path_exists(irr1, irr2, ir)]).simplify()
            self.nonlin = None

        self.tp = SeparateWeightTensorProduct(irr1, irr2, irr_tp_out)

        if self.nonlin_on:
            self.cfconv = E3ElementWise(str(self.nonlin.irreps_out))
            irr_after = self.nonlin.irreps_out
        else:
            self.cfconv = E3ElementWise(str(irr_tp_out))
            irr_after = irr_tp_out

        self.fc = nn.Sequential(
            nn.Linear(fc_len_in, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.cfconv.len_weight),
        )

        if self.use_norm:
            self.norm = e3LayerNorm(irr_after)
        else:
            self.norm = None

    def forward(self, fea_in1: torch.Tensor, fea_in2: torch.Tensor,
                fea_weight: Optional[torch.Tensor], batch_edge: torch.Tensor) -> torch.Tensor:
        z = self.tp(fea_in1, fea_in2)
        if self.nonlin_on and self.nonlin is not None:
            z = self.nonlin(z)
        if fea_weight is not None:
            w = self.fc(fea_weight)
            z = self.cfconv(z, w)
        if self.use_norm and self.norm is not None:
            z = self.norm(z, batch_edge)
        return z