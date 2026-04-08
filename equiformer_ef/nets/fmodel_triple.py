import torch
from torch import nn
from torch.autograd import grad
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode

import torch_geometric
import math

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .fast_layer_norm import EquivariantLayerNormFast
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .gaussian_rbf import GaussianRadialBasisLayer

# for bessel radial basis
# from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis

from .graph_attention_transformer import (get_norm_layer, 
    FullyConnectedTensorProductRescaleNorm, 
    FullyConnectedTensorProductRescaleNormSwishGate, 
    FullyConnectedTensorProductRescaleSwishGate,
    DepthwiseTensorProduct, SeparableFCTP,
    Vec2AttnHeads, AttnHeads2Vec,
    GraphAttention, FeedForwardNetwork, 
    TransBlock, 
    EdgeDegreeEmbeddingNetwork, ScaledScatter
)

from e3nn.o3 import Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
# from .model_jit import EquiConv
from tg_src.model import EquiConv
from tg_src.e3modules import e3LayerNorm


_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 128 # Set to some large value

# Statistics of QM9 with cutoff radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 144 #18.03065905448718
_AVG_DEGREE = 15.57930850982666



class Equi_Nonlin_Grad_Module(nn.Module):
    def __init__(self, irreps_in, z_fea_len=64, hidden_fea_len=64):
        super(Equi_Nonlin_Grad_Module, self).__init__()
        self.irreps_in = irreps_in
        self.fctp = FullyConnectedTensorProduct(irreps_in, irreps_in, f'{hidden_fea_len}x0e')
        self.nonlinear_layers = nn.Sequential(
            nn.Linear(hidden_fea_len, hidden_fea_len),
            # torch.nn.LayerNorm(hidden_fea_len, eps=1e-6),
            nn.SiLU(),
            nn.Linear(hidden_fea_len, z_fea_len),
            # torch.nn.LayerNorm(z_fea_len, eps=1e-6),
            nn.SiLU(),
            nn.Linear(z_fea_len, z_fea_len),
        )

    def forward(self, tensor_in,retain_graph=True):
        
        x = self.fctp(tensor_in, tensor_in)
        x = self.nonlinear_layers(x)
        y = grad(outputs=x, inputs=tensor_in, grad_outputs=torch.ones_like(x), retain_graph=retain_graph, create_graph=retain_graph, only_inputs=True, 
        allow_unused=True)[0]
        # y = torch.zeros_like(y)
        return x, y
    
def build_triplet_ij_ik(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    exclude_same_edge: bool = False,
    exclude_self_loop_k: bool = False,
):
    """
    构建 triple 索引 (ij, ik)，两条边共享同一个起点 i。
    使用纯 torch 张量并行，不使用 Python for 循环。

    约定
    ----
    edge_index[0] = i, edge_index[1] = j
    edge_vec[e] = r_ij = x_j - x_i

    对于 triple (ij, ik):
        rij = edge_vec[target_edge_id]
        rik = edge_vec[source_edge_id]
        rjk = rik - rij = x_k - x_j

    返回
    ----
    target_edge_id : LongTensor [T]
        目标边 e_ij 的编号
    source_edge_id : LongTensor [T]
        源边 e_ik 的编号
    rik_norm : Tensor [T]
        ||r_ik||
    rjk_norm : Tensor [T]
        ||r_jk||
    """
    device = edge_index.device
    edge_src, edge_dst = edge_index
    E = edge_src.numel()

    if E == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = edge_vec.new_empty(0)
        return empty_long, empty_long, empty_float, empty_float

    # 1) 按 src 排序，把同一个 i 的所有边放到连续区间
    perm = torch.argsort(edge_src)
    src_sorted = edge_src[perm]

    # 2) 每个组的大小 counts，以及每组在排序后数组中的起始位置 starts
    _, counts = torch.unique_consecutive(src_sorted, return_counts=True)
    starts = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        counts.cumsum(dim=0)[:-1]
    ])  # [G]

    # 3) 每组会产生 m*m 个有序 pair
    pair_counts = counts * counts                  # [G]
    total_pairs = int(pair_counts.sum().item())

    if total_pairs == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = edge_vec.new_empty(0)
        return empty_long, empty_long, empty_float, empty_float

    # 4) 对每个 pair，标记它属于哪个组
    group_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device),
        pair_counts
    )  # [P]

    # 5) 组内 pair 的线性编号 0..m*m-1
    pair_offsets = torch.repeat_interleave(
        torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            pair_counts.cumsum(dim=0)[:-1]
        ]),
        pair_counts
    )
    local_pair_id = torch.arange(total_pairs, device=device) - pair_offsets  # [P]

    # 6) 将线性编号拆成组内 (a, b)
    #    a: target 在组内的位置
    #    b: source 在组内的位置
    m = counts[group_id]                      # [P]
    a = torch.div(local_pair_id, m, rounding_mode='floor')
    b = local_pair_id % m

    # 7) 转成排序后边数组中的位置，再映射回原始边编号
    tgt_pos = starts[group_id] + a
    src_pos = starts[group_id] + b

    target_edge_id = perm[tgt_pos]           # e_ij
    source_edge_id = perm[src_pos]           # e_ik

    # 8) 过滤无效 pair
    mask = torch.ones(total_pairs, dtype=torch.bool, device=device)

    if exclude_same_edge:
        mask &= target_edge_id != source_edge_id

    if exclude_self_loop_k:
        i = edge_src[target_edge_id]
        k = edge_dst[source_edge_id]
        mask &= (k != i)

    target_edge_id = target_edge_id[mask]
    source_edge_id = source_edge_id[mask]

    # 9) 几何量
    rij = edge_vec[target_edge_id]           # [T, 3]
    rik = edge_vec[source_edge_id]           # [T, 3]
    rjk = rik - rij                          # [T, 3]

    rik_norm = rik.norm(dim=-1)              # [T]
    rjk_norm = rjk.norm(dim=-1)              # [T]

    return target_edge_id, source_edge_id, rik_norm, rjk_norm

@torch.no_grad()
def get_inverse_edge_index(edge_src, edge_dst, edge_vec, atol=1e-6, rtol=1e-5):
    """
    对每条边 e: i -> j，找到唯一的反向边 e_inv，要求同时满足：
        1) edge_src[e_inv] == edge_dst[e]
        2) edge_dst[e_inv] == edge_src[e]
        3) edge_vec[e_inv] == -edge_vec[e]   (在 atol/rtol 误差内)

    参数
    ----
    edge_src : LongTensor [E]
    edge_dst : LongTensor [E]
    edge_vec : Tensor [E, D]
    atol, rtol : float
        用于判断 edge_vec 是否互为相反数

    返回
    ----
    inverse_index : LongTensor [E]
        inverse_index[e] 是边 e 的唯一反向边编号
    """
    device = edge_src.device
    E = edge_src.shape[0]

    row = edge_src.long()
    col = edge_dst.long()

    num_nodes = int(torch.max(torch.cat([row, col])).item()) + 1

    key = row * num_nodes + col
    rev_key = col * num_nodes + row

    perm = torch.argsort(key)
    key_sorted = key[perm]

    # 每条边先找到所有 (j, i) 候选所在的连续区间
    left = torch.searchsorted(key_sorted, rev_key, right=False)
    right = torch.searchsorted(key_sorted, rev_key, right=True)

    counts = right - left
    assert torch.all(counts > 0), "存在边连 (j, i) 方向的候选边都没有"

    # 收集所有候选匹配 (e, cand_e)
    edge_ids = torch.repeat_interleave(torch.arange(E, device=device), counts)
    cand_pos = torch.arange(counts.sum(), device=device) - torch.repeat_interleave(
        torch.cumsum(counts, dim=0) - counts, counts
    ) + torch.repeat_interleave(left, counts)
    cand_index = perm[cand_pos]

    # 检查 edge_vec 是否互为相反数
    vec_ok = torch.all(
        torch.isclose(edge_vec[cand_index], -edge_vec[edge_ids], atol=atol, rtol=rtol),
        dim=-1
    )

    # 每条边满足条件的候选个数必须恰好为 1
    valid_count = torch.zeros(E, dtype=torch.long, device=device)
    valid_count.scatter_add_(0, edge_ids, vec_ok.long())
    assert torch.all(valid_count == 1), (
        "存在边不满足“恰好一条反向边”条件："
        "要么没有满足 edge_vec 相反条件的候选，要么有多条满足条件的候选"
    )

    # 取出唯一合法候选
    valid_edge_ids = edge_ids[vec_ok]
    valid_cand_index = cand_index[vec_ok]

    inverse_index = torch.empty(E, dtype=torch.long, device=device)
    inverse_index[valid_edge_ids] = valid_cand_index

    # 双向一致性检查
    arange = torch.arange(E, device=device)
    assert torch.equal(inverse_index[inverse_index], arange), "反向边映射不是对合映射"
    assert torch.all(inverse_index != arange), "存在边把自己匹配成反向边"

    return inverse_index

class Block(torch.nn.Module):
    def __init__(
        self,
        irreps_node_in,
        irreps_edge,
        irreps_node_out,
        irreps_sh,
        num_heads,
        edge_length_len,
        out_layer = False
    ):
        super().__init__()
        self.irreps_node_in = Irreps(irreps_node_in)
        self.irreps_edge = Irreps(irreps_edge)
        self.irreps_node_out = Irreps(irreps_node_out)
        self.irreps_sh = Irreps(irreps_sh)
        self.num_heads = num_heads
        self.out_layer = out_layer

        self.pre_lin = Linear(
            self.irreps_node_in + self.irreps_node_in + Irreps(f"{edge_length_len}x0e"),
            self.irreps_node_in,
        )
        self.tp = EquiConv(edge_length_len, self.irreps_node_in, self.irreps_sh, self.irreps_node_in)
        self.tp2 = EquiConv(
            edge_length_len,
            self.irreps_node_in,
            self.irreps_sh,
            self.irreps_node_in * self.num_heads,
        )


        self.lin_alpha1 = torch.nn.Sequential(
            torch.nn.Linear(edge_length_len, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.num_heads),
        )

        self.lin_alpha2 = torch.nn.Sequential(
            torch.nn.Linear(edge_length_len, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.num_heads),
        )


        self.lin = Linear(self.irreps_node_in * self.num_heads, self.irreps_node_out)

        self.lin_edge = Linear(self.irreps_node_in * self.num_heads, Irreps("64x0e"))
        self.lin_scalar = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32),
        )

    def forward(self, edge_in, edge_in_inverse_index, edge_sh, edge_length_embedding, edge_vec, edge_src, edge_dst, triple_fea, batch):
        message = torch.cat(
            (edge_in, edge_in[edge_in_inverse_index], edge_length_embedding),
            dim=-1,
        )
        message = self.pre_lin(message)

        value = self.tp2(edge_in, edge_sh, edge_length_embedding, batch[edge_src])
        value = value.reshape(value.shape[0], self.num_heads, -1)

        value = value[edge_in_inverse_index][triple_fea[1]]
        alpha = self.lin_alpha1(triple_fea[2])*self.lin_alpha2(triple_fea[3])
        alpha = torch_geometric.utils.softmax(alpha, triple_fea[0]).unsqueeze(-1)

        edge_fea = scatter(value * alpha.unsqueeze(-1), index=triple_fea[0], dim=0)
        edge_out = self.lin(edge_fea.reshape(edge_fea.shape[0], -1))
        out = edge_out
        if self.out_layer:
            node_out = scatter(edge_out, index=edge_dst, dim=0)
            out = node_out


        return out





class Nets(torch.nn.Module):
    def __init__(self,
        irreps_in='64x0e',
        irreps_node_embedding='128x0e+64x1o+32x2e', irreps_edge_embedding = '32x0e+16x1o+8x2e+4x3o+4x4e',
        irreps_edge_output = '1x0e+1x0e+1x1o+1x1o+1x2e+1x0e+1x0e+1x1o+1x1o+1x2e+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x2e+1x2e+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x0e+1x1e+1x2e+1x3e+1x4e', 
        num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=6.0,
        number_of_basis=128, fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        use_attn_head=False, 
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, std=None, scale=None, atomref=None
        ):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.use_attn_head = use_attn_head
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref) 

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.irreps_edge_mid = o3.Irreps(irreps_edge_embedding)
        self.irreps_edge_output = o3.Irreps(irreps_edge_output)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        
        self.rbf = GaussianExpansion(0, 6, num_basis=self.number_of_basis)
        # self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=10)
        
        self.edge_prelin = Linear(self.irreps_edge_attr, self.irreps_edge_mid)
        # self.edge_lin = LinearRS(o3.Irreps(f'32x0e+{self.irreps_edge_output.sort()[0].simplify()}'), self.irreps_edge_output)

        # self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1)
        )

        self.e0_lin = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            # torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            # torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1)
            )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.blocks = torch.nn.ModuleList()

        self.base_energy = None

        self.edge_scalar_out = False
        if self.edge_scalar_out:
            self.edge_scalar_out_len = 32
        else:
            self.edge_scalar_out_len = 0
        residual = True
        for i in range(num_layers):
            if i < num_layers - 1:
                block = Block(self.irreps_node_embedding, self.irreps_edge_mid, 
                              self.irreps_node_embedding, self.irreps_sh, num_heads, 
                              edge_length_len = self.number_of_basis + self.edge_scalar_out_len, )
            else:
                block = Block(self.irreps_node_embedding, self.irreps_edge_mid, 
                              self.irreps_feature, self.irreps_sh, num_heads,
                              edge_length_len = self.number_of_basis + self.edge_scalar_out_len, out_layer = True)

            self.blocks.append(block)

    @torch.enable_grad()
    def forward(self, node_atom, edge_src, edge_dst, edge_vec, batch, R = torch.eye(3), **kwargs):
        S = torch.eye(3, requires_grad = True, device = edge_vec.device, dtype = edge_vec.dtype)
        edge_vec = edge_vec.requires_grad_(True) #@ S
        
        # import pdb; pdb.set_trace()
        edge_vec = (edge_vec[:, [1,2,0]] @ o3.Irreps("1x1o").D_from_matrix(R).to(edge_vec.device))[:,[2,0,1]]
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec[:, [1,2,0]], normalize=True, normalization='component')
        
        node_features, atom_attr, atom_onehot = self.atom_embed(node_atom)

        e0 = self.e0_lin(node_features[:, :64])
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)

        source_edge_id, target_edge_id, lik, ljk = build_triplet_ij_ik(torch.stack((edge_src, edge_dst), dim=0), edge_vec)

        lik = edge_length_embedding[source_edge_id]
        ljk = self.rbf(ljk)

        triple_fea = (source_edge_id, target_edge_id, lik, ljk)

        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        edge_features = self.edge_prelin(edge_sh)
        if self.edge_scalar_out:
            edge_scalar = torch.cat((edge_length_embedding, 
                                    torch.zeros_like(edge_length_embedding[:, :self.edge_scalar_out_len])), 
                                    dim = -1)
        else:
            edge_scalar = edge_length_embedding
        
        edge_features_invar_list = []
        node_embed = node_features
        out_features = torch.cat(node_features[edge_src])
        edge_inverse_index = get_inverse_edge_index(edge_src, edge_dst, edge_vec)

        for blk_idx, blk in enumerate(self.blocks):
            out_features = blk(edge_in = out_features, edge_in_inverse_index = edge_inverse_index, edge_sh = edge_sh, 
                                edge_length_embedding = edge_scalar, edge_vec = edge_vec, 
                                edge_src = edge_src, edge_dst = edge_dst, triple_fea = triple_fea, batch = batch)
        

        # energy = self.head(node_features)[:, 0] + e0[:, 0] + atom_energy[node_atom]
        base_energy = self.base_energy.to(device=node_atom.device)
        energy = self.head(out_features)[:, 0] + e0[:, 0] + base_energy[node_atom]
        # energy = self.head(node_features)[:, 0]

        energy = scatter(energy, batch, dim=0)

        mask = edge_vec.norm(dim = -1) != 0
        force_edge = torch.autograd.grad(energy.sum(), edge_vec, create_graph=True, only_inputs=True, allow_unused=True)[0]#[mask]
        forces = torch.zeros(node_atom.shape[0], 3, device = edge_vec.device, dtype = edge_vec.dtype)
        filtered_edge_index_0 = edge_src[mask]
        filtered_edge_index_1 = edge_dst[mask]
        filtered_force_edge = force_edge
        forces.index_add_(0, filtered_edge_index_1, -filtered_force_edge)
        forces.index_add_(0, filtered_edge_index_0, filtered_force_edge)
        
        virial = -edge_vec[mask].unsqueeze(2) * force_edge[mask].unsqueeze(1)
        virial1 = scatter(virial, batch[edge_src], dim = 0)
        # virial2 = torch.autograd.grad(energy.sum(), S, create_graph=True, only_inputs=True, allow_unused=True)[0]

        return energy, (forces[:, [1,2,0]] @ o3.Irreps("1x1o").D_from_matrix(R.T).to(edge_vec.device))[:,[2,0,1]], virial1

@register_model
def f_model(irreps_in, irreps_edge, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, with_trace = True, trace_out_len=25, start_layer=0, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge, 
        irreps_node_embedding='64x0e+32x1o+16x2e', 
        irreps_edge_embedding='64x0e+32x1o+16x2e', num_layers=3,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=8.0,
        number_of_basis=64, fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='64x0e+32x1o+16x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='64x0e+32x1o+16x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
        )
    return model

@register_model
def f_model_large(irreps_in, irreps_edge, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, with_trace = True, trace_out_len=25, start_layer=0, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge, 
        irreps_node_embedding='64x0e+32x1o+32x1e+16x2e', 
        irreps_edge_embedding='64x0e+32x1o+32x1e+16x2e', num_layers=3,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=8.0,
        number_of_basis=64, fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='64x0e+32x1o+32x1e+16x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='64x0e+32x1o+32x1e+16x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
        )
    return model

@register_model
def my_model_complex(irreps_in, irreps_edge, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, with_trace = True, trace_out_len=25, start_layer=0, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge, 
        irreps_node_embedding='32x0e+16x1o+8x2e+8x3o+8x4e', 
        irreps_edge_embedding='32x0e+16x1o+8x2e+8x3o+8x4e', num_layers=4,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e+1x3o+1x4e',
        max_radius=8.0,
        number_of_basis=64, fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e+8x3o+8x4e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='32x0e+16x1o+8x2e+8x3o+8x4e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
        )
    return model

class GaussianExpansion(torch.nn.Module):
    def __init__(self, min_val, max_val, num_basis):
        super(GaussianExpansion, self).__init__()
        self.register_buffer('centers', torch.linspace(min_val, max_val, num_basis, dtype=torch.float32))
        self.width = 0.5 * (max_val - min_val) / num_basis

    def forward(self, x):
        x = x.unsqueeze(-1)
        centers = self.centers.to(device=x.device, dtype=x.dtype)
        return torch.exp(-((x - centers) ** 2) / (2 * self.width ** 2))
    

class NodeEmbeddingNetwork(torch.nn.Module):
    
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)), 
            self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)
        
        
    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)
        
        return node_embedding, node_attr, node_atom_onehot
