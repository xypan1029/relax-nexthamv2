import warnings
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from torch_scatter import scatter
from torch_geometric.utils import degree
try:
    from .SO2_tools import SO2_Convolution, SO2_Convolution_ParitySplit
except ImportError:
    from SO2_tools import SO2_Convolution, SO2_Convolution_ParitySplit


class e3LayerNorm(nn.Module):
    def __init__(self, irreps_in, eps=1e-5, affine=True, normalization='component', subtract_mean=True, divide_norm=False):
        super().__init__()
        
        self.irreps_in = Irreps(irreps_in)
        self.eps = eps
        
        if affine:          
            ib, iw = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar(): # bias only to 0e
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        assert normalization in ['component', 'norm']
        self.normalization = normalization
            
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.weight is not None:
            self.weight.data.fill_(1)
            # nn.init.uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)
            # nn.init.uniform_(self.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        # input x must have shape [num_node(edge), dim]
        # if first dimension of x is node index, then batch should be batch.batch
        # if first dimension of x is edge index, then batch should be batch.batch[batch.edge_index[0]]
        
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64)

        # from torch_geometric.nn.norm.LayerNorm

        batch_size = int(batch.max()) + 1 
        batch_degree = degree(batch, batch_size, dtype=torch.int64).clamp_(min=1).to(dtype=x.dtype)
        
        out = []
        ix = 0
        for index, (mul, ir) in enumerate(self.irreps_in):        
            field = x[:, ix: ix + mul * ir.dim].reshape(-1, mul, ir.dim) # [node, mul, repr]
            
            # compute and subtract mean
            if self.subtract_mean or ir.l == 0: # do not subtract mean for l>0 irreps if subtract_mean=False
                mean = scatter(field, batch, dim=0, dim_size=batch_size,
                            reduce='add').mean(dim=1, keepdim=True) / batch_degree[:, None, None] # scatter_mean does not support complex number
                field = field - mean[batch]
                
            # compute and divide norm
            if self.divide_norm or ir.l == 0: # do not divide norm for l>0 irreps if subtract_mean=False
                norm = scatter(field.abs().pow(2), batch, dim=0, dim_size=batch_size,
                            reduce='mean').mean(dim=[1,2], keepdim=True) # add abs here to deal with complex numbers
                if self.normalization == 'norm':
                    norm = norm * ir.dim
                field = field / (norm.sqrt()[batch] + self.eps)
            
            # affine
            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]
            
            out.append(field.reshape(-1, mul * ir.dim))
            ix += mul * ir.dim
            
        out = torch.cat(out, dim=-1)
                
        return out


class E3ElementWise(nn.Module):
    muls: List[int]
    dims: List[int]
    len_weight: int
    out_dim: int

    def __init__(self, irreps_in: str):
        super().__init__()
        ir = o3.Irreps(irreps_in)  # 只在 __init__ 用，不存为属性（避免持有复杂Python对象）
        muls, dims = [], []
        for mul, irrep in ir:
            muls.append(int(mul))
            dims.append(int(irrep.dim))
        # 这些是纯 Python 基本类型/列表，TorchScript 支持作为属性（已在类体注解）
        self.muls = muls
        self.dims = dims
        self.len_weight = int(sum(muls))
        self.out_dim = int(sum(m * d for m, d in zip(muls, dims)))

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x: [N, out_dim], weight: [N, len_weight]
        N = x.size(0)
        ix = 0
        iw = 0
        outs: List[torch.Tensor] = []
        # 只在图里操作 Tensor / list[Tensor]
        for i in range(len(self.muls)):
            mul_i = self.muls[i]
            dim_i = self.dims[i]
            span = mul_i * dim_i

            field = x[:, ix: ix + span].reshape(N, mul_i, dim_i)
            w = weight[:, iw: iw + mul_i].unsqueeze(-1)  # [N, mul, 1]
            field = field * w                             # 逐“拷贝”尺度化

            outs.append(field.reshape(N, span))
            ix += span
            iw += mul_i

        return torch.cat(outs, dim=-1)

class GateWithControl(nn.Module): 
    """
    将输入张量 x 的 l=0 作为 scalars，l>0 作为 gated；
    把 [x 的 l=0 标量 | 外部 gate_in 标量] 拼接后过一个 MLP 产生 gates
    （gates 的个数 = 所有 l>0 的 multiplicity 之和），
    构成 irreps_gates = [(mul, 0e) for mul,_ in irreps_gated].simplify() 的 gates，
    再交给 e3nn.nn.Gate 做门控。
    """
    def __init__(
        self,
        irreps_in,        # 输入张量的 irreps（按 e3nn 标准排列）
        gate_in_dim,      # 外部 gate_in 的长度（任意），会与 x 的 l=0 拼接后送入 MLP
        act={1: torch.nn.functional.silu, -1: torch.tanh},
        act_gates={1: torch.sigmoid, -1: torch.tanh},
        mlp_hidden: int = 128,         # MLP 隐藏维度
        mlp_layers: int = 2,           # MLP 层数（>=1）
        dropout: float = 0.0,          # MLP dropout
        use_bias: bool = True          # MLP 线性层是否带 bias
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in).simplify()
        self.gate_in_dim = int(gate_in_dim)
        self.act = act
        self.act_gates = act_gates
        self.mlp_hidden = int(mlp_hidden)
        self.mlp_layers = int(mlp_layers)
        self.dropout = float(dropout)
        self.use_bias = bool(use_bias)

        # 拆分 l=0 / l>0
        self.irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l == 0]).simplify()
        self.irreps_gated   = Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l > 0]).simplify()

        # 需要的 gate 个数 = 所有 l>0 的 multiplicity 总和
        # 对 Gate 的规范：irreps_gates == [(mul, "0e") for (mul, _) in irreps_gated].simplify()
        self.gates_per_block = [mul for (mul, _ir) in self.irreps_gated]  # 按块顺序
        self.num_gates_needed = sum(self.gates_per_block)
        if self.num_gates_needed > 0:
            assert (self.gate_in_dim >= 0), "gate_in_dim 必须 >= 0"
            self.irreps_gates = Irreps([(mul, "0e") for mul, _ in self.irreps_gated]).simplify()
        else:
            self.irreps_gates = Irreps([])

        # 构造 Gate
        self.gate = Gate(
            self.irreps_scalars, [self.act[ir.p] for _, ir in self.irreps_scalars],
            self.irreps_gates,   [self.act_gates[ir.p] for _, ir in self.irreps_gates],
            self.irreps_gated
        )
        self.irreps_out = self.gate.irreps_out  # 方便外部查询

        # 记录维度，便于检查
        self.dim_in = self.irreps_in.dim
        self.dim_scalars = self.irreps_scalars.dim
        self.dim_gated   = self.irreps_gated.dim

        # ------- 索引缓存：把 x 拆成 scalars / gated 用 -------
        scal_idx, gated_idx = [], []
        cur = 0
        for mul, ir in self.irreps_in:
            span = mul * ir.dim
            idxs = list(range(cur, cur + span))
            if ir.l == 0:
                scal_idx.extend(idxs)
            else:
                gated_idx.extend(idxs)
            cur += span

        self.register_buffer(
            "scal_idx",
            torch.tensor(scal_idx, dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "gated_idx",
            torch.tensor(gated_idx, dtype=torch.long),
            persistent=False
        )

        # ------- MLP: [ scalars(x) | gate_in ] -> gates -------
        # 当 num_gates_needed == 0 时不构建 MLP
        if self.num_gates_needed > 0:
            mlp_in_dim = self.dim_scalars + self.gate_in_dim  # 把 x 的 l=0 和 gate_in 拼起来
            layers = []
            if self.mlp_layers <= 1:
                # 单层线性 + LayerNorm
                layers.append(nn.Linear(mlp_in_dim, self.num_gates_needed, bias=self.use_bias))
                layers.append(nn.LayerNorm(self.num_gates_needed))
            else:
                # 首层
                layers.append(nn.Linear(mlp_in_dim, self.mlp_hidden, bias=self.use_bias))
                layers.append(nn.LayerNorm(self.mlp_hidden))
                layers.append(nn.SiLU())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                # 中间隐藏层（mlp_layers-2 个）
                for _ in range(self.mlp_layers - 2):
                    layers.append(nn.Linear(self.mlp_hidden, self.mlp_hidden, bias=self.use_bias))
                    layers.append(nn.LayerNorm(self.mlp_hidden))
                    layers.append(nn.SiLU())
                    if self.dropout > 0:
                        layers.append(nn.Dropout(self.dropout))
                # 末层到 gates
                layers.append(nn.Linear(self.mlp_hidden, self.num_gates_needed, bias=self.use_bias))
                layers.append(nn.LayerNorm(self.num_gates_needed))
            self.gates_mlp = nn.Sequential(*layers)
        else:
            self.gates_mlp = None  # 不会用到


    # 拆分 x → (scalars, gated)
    def _split_scalars_gated(self, x: torch.Tensor):
        N, D = x.shape
        assert D == self.dim_in, f"x.shape[1]={D} 与 irreps_in.dim={self.dim_in} 不符"
        scalars = (
            x.index_select(1, self.scal_idx)
            if self.dim_scalars > 0 else
            x.new_zeros((N, 0), dtype=x.dtype, device=x.device)
        )
        gated = (
            x.index_select(1, self.gated_idx)
            if self.dim_gated > 0 else
            x.new_zeros((N, 0), dtype=x.dtype, device=x.device)
        )
        return scalars, gated

    # 打包 Gate 的输入：[scalars | gates | gated]
    def _pack_inputs(self, scalars, gates_mapped, gated):
        if self.num_gates_needed > 0:
            assert gates_mapped is not None and gates_mapped.shape[0] == scalars.shape[0] \
                   and gates_mapped.shape[1] == self.num_gates_needed, \
                   f"gates_mapped 需要形状 [N,{self.num_gates_needed}]，得到 {tuple(gates_mapped.shape)}"
            X = torch.cat([scalars, gates_mapped, gated], dim=1)
        else:
            X = torch.cat([scalars, gated], dim=1)
        return X

    def forward(self, x, gate_in):
        """
        x:       [N, irreps_in.dim]  （l=0 与 l>0 混合）
        gate_in: [N, gate_in_dim]    （外部标量，和 x 的 l=0 拼接后送入 MLP 生成 gates）
        返回：Gate 输出张量（按 Gate 的输出 irreps 排列）
        """
        assert x.dim() == 2 and x.size(1) == self.dim_in, \
            f"x.shape[1]={x.size(1)} 与 irreps_in.dim={self.dim_in} 不一致"

        scalars, gated = self._split_scalars_gated(x)  # 从 x 拆出 l=0 / l>0

        # 生成 gates：gates = MLP( concat( scalars(x), gate_in ) )
        if self.num_gates_needed > 0:
            if self.gate_in_dim > 0:
                assert gate_in is not None and gate_in.dim() == 2 \
                       and gate_in.size(0) == x.size(0) and gate_in.size(1) == self.gate_in_dim, \
                       f"gate_in 需为 [N,{self.gate_in_dim}]，得到 {tuple(gate_in.shape)}"
                mlp_in = torch.cat([scalars, gate_in], dim=1)
            else:
                # 若 gate_in_dim=0，则仅用 scalars(x) 生成 gates
                mlp_in = scalars
            gates_mapped = self.gates_mlp(mlp_in)  # [N, num_gates_needed]
        else:
            gates_mapped = None

        # 打包并进入 Gate
        X = self._pack_inputs(scalars, gates_mapped, gated)
        y = self.gate(X)
        return y


class SO2_EquiConv(nn.Module):
    def __init__(self, fc_len_in, irreps_in, irreps_out, norm='', nonlin=True, cfconv = True, 
                 act = {1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates = {1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(SO2_EquiConv, self).__init__()
        
        # assert nonlin is True
        irreps_in1 = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        
        gate_dim = 64

        self.nonlin = GateWithControl(
            irreps_in = irreps_out,
            gate_in_dim = fc_len_in,
            act = act,
            act_gates = act_gates
        )

        self.tp = SO2_Convolution_ParitySplit(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            fea_weight_dim=fc_len_in,
            gate_dim=gate_dim,
            mlp_hidden=64)
        
        self.cfconv = None
        if cfconv:
            self.cfconv = E3ElementWise(self.nonlin.irreps_out)
        
        
            # fully connected net to create tensor product weights
            linear_act = nn.SiLU()
            self.fc = nn.Sequential(nn.Linear(fc_len_in, 64),
                                    torch.nn.LayerNorm(64, eps=1e-6),
                                    linear_act,
                                    nn.Linear(64, self.cfconv.len_weight)
                                    )

        self.irreps_out = irreps_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.nonlin.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')

        if not nonlin:
            self.nonlin = None

    def forward(self, fea_edge, edge_vec, D_dict, fea_weight, batch_edge):
        z, gate_in = self.tp(fea_edge, edge_vec, D_dict, fea_weight)

        if self.nonlin is not None:
            z = self.nonlin(z, fea_weight)
        if fea_weight is not None and self.cfconv is not None:
            weight = self.fc(fea_weight)
            z = self.cfconv(z, weight)
        
        if self.norm is not None:
            z = self.norm(z, batch_edge.to(torch.int64))

        # TODO self-connection here
        return z