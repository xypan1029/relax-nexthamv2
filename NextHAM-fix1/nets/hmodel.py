import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists
from .layer_norm import EquivariantLayerNormV2

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
from .tracegrad import Equi_Nonlin_Grad_Module

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
    NodeEmbeddingNetwork, EdgeDegreeEmbeddingNetwork, ScaledScatter
)

from e3nn.o3 import Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from tg_src.model import Tp_nonlin, EquiConv
from tg_src.e3modules import e3LayerNorm
from .SO2_conv import SO2_EquiConv
from .SO3_tools import compute_D


_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 128 # Set to some large value

# Statistics of QM9 with cutoff radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 144 #18.03065905448718
_AVG_DEGREE = 15.57930850982666

class Block(torch.nn.Module):
    def __init__(
        self,
        irreps_node_in,
        irreps_head,
        irreps_edge_in,
        irreps_edge_out,
        irreps_node_out,
        irreps_sh,
        num_heads,
        qk_head_dim=16,
    ):
        super().__init__()
        self.irreps_node_in = Irreps(irreps_node_in)
        self.irreps_head = Irreps(irreps_head)
        self.irreps_edge_out = Irreps(irreps_edge_out)
        self.irreps_node_out = Irreps(irreps_node_out)
        self.irreps_edge_in = Irreps(irreps_edge_in)
        self.irreps_sh = Irreps(irreps_sh)
        self.num_heads = num_heads
        self.qk_head_dim = qk_head_dim

        # pre-norm
        self.norm_edge_in = EquivariantLayerNormV2(self.irreps_edge_in)
        self.norm_node_in = EquivariantLayerNormV2(self.irreps_node_in)


        self.pre_lin = Linear(
            self.irreps_node_in + self.irreps_node_in + self.irreps_edge_in,
            self.irreps_node_in
        )

        irreps_attn_heads, _, _ = sort_irreps_even_first(self.irreps_head * self.num_heads)
        irreps_attn_heads = irreps_attn_heads.simplify()

        self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        self.heads2vec = AttnHeads2Vec(irreps_head)
        self.heads2vec_edge = AttnHeads2Vec(irreps_head)

        self.lin = Linear(irreps_attn_heads, self.irreps_node_out)
        self.lin_edge = Linear(irreps_attn_heads, self.irreps_edge_out)

        self.SO2_conv1 = SO2_EquiConv(
            64,
            self.irreps_node_in,
            self.irreps_node_in,
            norm='e3LayerNorm',
            nonlin=True
        )
        self.SO2_conv2 = SO2_EquiConv(
            64,
            self.irreps_node_in,
            irreps_attn_heads,
            norm="",
            nonlin=False,
            cfconv=False
        )
        self.onsite_lin = LinearRS(self.irreps_node_in, irreps_attn_heads)

        self.node_in_slices = self.irreps_node_in.slices()
        self.scalar_slices = []
        scalar_dim = 0

        for mul_ir, sl in zip(self.irreps_node_in, self.node_in_slices):
            if mul_ir.ir.l == 0 and mul_ir.ir.p == 1:
                self.scalar_slices.append(sl)
                scalar_dim += (sl.stop - sl.start)

        self.scalar_dim = scalar_dim

        # scalar
        self.q_proj = torch.nn.Sequential(
            torch.nn.Linear(self.scalar_dim, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.num_heads * self.qk_head_dim)
        )
        self.k_proj = torch.nn.Sequential(
            torch.nn.Linear(self.scalar_dim, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.num_heads * self.qk_head_dim)
        )

        self.lin_alpha = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.num_heads)
        )

        # ---------------------------
        # residual / skip projection
        # ---------------------------
        self.node_skip = Linear(self.irreps_node_in, self.irreps_node_out)
        self.edge_skip = Linear(self.irreps_edge_in, self.irreps_edge_out)

        self.ffn_placeholder = None

    def _extract_scalar_part(self, x):
        """
        x: [N, dim(irreps_node_in)]
        return: [N, scalar_dim]
        """
        scalar_parts = [x[:, sl] for sl in self.scalar_slices]
        if len(scalar_parts) == 1:
            return scalar_parts[0]
        return torch.cat(scalar_parts, dim=-1)

    def forward(
        self,
        node_fea,
        edge_fea,
        edge_sh,
        edge_length_embedding,
        edge_src,
        edge_dst,
        edge_vec,
        D_dict,
        batch
    ):
        node_fea_res = node_fea
        edge_fea_res = edge_fea

        node_fea = self.norm_node_in(node_fea)
        edge_fea = self.norm_edge_in(edge_fea)

        node_scalar = self._extract_scalar_part(node_fea)   # [N, scalar_dim]

        q = self.q_proj(node_scalar[edge_dst])   # [E, H * D]
        k = self.k_proj(node_scalar[edge_src])   # [E, H * D]

        q = q.view(-1, self.num_heads, self.qk_head_dim)   # [E, H, D]
        k = k.view(-1, self.num_heads, self.qk_head_dim)   # [E, H, D]

        qk_score = (q * k).sum(dim=-1) / math.sqrt(self.qk_head_dim)   # [E, H]

        edge_bias = self.lin_alpha(edge_length_embedding)   # [E, H]

        alpha = qk_score + edge_bias   # [E, H]

        message = torch.cat((node_fea[edge_src], node_fea[edge_dst], edge_fea), dim=-1)
        message = self.pre_lin(message)

        onsite = edge_vec.norm(dim=-1) < 1e-10

        value = self.SO2_conv1(message, edge_vec, D_dict, edge_length_embedding, batch[edge_src])
        value = self.SO2_conv2(value, edge_vec, D_dict, edge_length_embedding, batch[edge_src])

        value[onsite] = self.onsite_lin(message[onsite])

        value = self.vec2heads_value(value)   # [E, H, ...head_dim...]

        edge_msg = value * alpha.unsqueeze(-1)

        alpha = torch_geometric.utils.softmax(alpha, edge_dst).unsqueeze(-1)  # [E, H, 1]
        node_msg = scatter(edge_msg, index=edge_dst, dim=0, dim_size=node_fea.shape[0])

        node_msg = self.heads2vec(node_msg)
        edge_msg = self.heads2vec_edge(edge_msg)

        node_out = self.lin(node_msg)
        edge_out = self.lin_edge(edge_msg)

        node_out = node_out + self.node_skip(node_fea_res)
        edge_out = edge_out + self.edge_skip(edge_fea_res)

        # ---------------------------------
        # FFN placeholder
        #   node_out = node_out + FFN(norm(node_out))
        # ---------------------------------
        # TODO: add FFN here

        return node_out, edge_out




class Nets(torch.nn.Module):
    def __init__(self,
        irreps_in='64x0e',
        irreps_node_embedding='128x0e+64x1o+32x2e', irreps_edge_embedding = '32x0e+16x1o+8x2e+4x3o+4x4e',
        irreps_edge_output = '1x0e+1x0e+1x1o+1x1o+1x2e+1x0e+1x0e+1x1o+1x1o+1x2e+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x2e+1x2e+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x0e+1x1e+1x2e+1x3e+1x4e', 
        num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=6.0,
        number_of_basis=128, 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        use_attn_head=False, 
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, std=None, scale=None, atomref=None, trace_out_len = 81
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
        # print(self.irreps_edge_output)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        
        self.rbf = GaussianExpansion(0, 10, num_basis=number_of_basis)
        
        self.edge_prelin = LinearRS(self.irreps_edge_attr, self.irreps_edge_mid)
        # self.edge_lin = LinearRS(o3.Irreps(f'32x0e+{self.irreps_edge_output.sort()[0].simplify()}'), self.irreps_edge_output)

        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            # torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            # torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1)
        )

        self.input_rs = LinearRS(self.irreps_edge_output, self.irreps_node_embedding)
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.blocks = torch.nn.ModuleList()

        for i in range(num_layers):
            if i < num_layers - 1:
                block = Block(self.irreps_node_embedding, self.irreps_head,
                                  self.irreps_node_embedding, self.irreps_node_embedding, 
                                  self.irreps_node_embedding, self.irreps_sh, num_heads)
            else:
                block = Block(self.irreps_node_embedding, self.irreps_head, 
                              self.irreps_node_embedding, self.irreps_edge_output.sort()[0].simplify(), 
                              self.irreps_feature, self.irreps_sh, num_heads)
            self.blocks.append(block)

        self.lin_out = Linear(self.irreps_edge_output.sort()[0].simplify(), self.irreps_edge_output)

    def forward(self, weak_ham_in, node_num, edge_src, edge_dst, edge_vec, batch, node_atom, use_sep = True, range_dis = [0.0, 10.0]):
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec[:, [1,2,0]], normalize=True, normalization='component')
        node_features, atom_attr, atom_onehot = self.atom_embed(node_atom)

        edge_length = edge_vec.norm(dim=1)

        edge_features = self.input_rs(weak_ham_in) 
        edge_length_embedding = self.rbf(edge_length)
        mask = torch.logical_and(edge_src == edge_dst, edge_length < 1e-5)
        filtered_edge_indices = edge_src[mask]
        filtered_edge_features = edge_features[mask]
        node_features = scatter(filtered_edge_features, filtered_edge_indices, dim=0, dim_size=node_num)
        
        
        mask_num = torch.logical_and(edge_length >= range_dis[0], edge_length < range_dis[1]).float().unsqueeze(1)
        
        # edge_features = edge_length_embedding
        edge_features0 = edge_features
        
        edge_features_invar_list = []
        D_dict = compute_D(edge_vec, l_max=7)
        for blk_idx, blk in enumerate(self.blocks):
            node_features, edge_features = blk(node_fea = node_features, edge_fea = edge_features0, edge_sh = edge_sh, 
                                edge_length_embedding = edge_length_embedding, 
                                edge_src = edge_src, edge_dst = edge_dst, edge_vec = edge_vec, D_dict = D_dict, batch = batch)
            
        edge_features = self.lin_out(edge_features) * mask_num 

        return edge_features, torch.zeros_like(edge_features), mask_num

@register_model
def h_model(irreps_in, irreps_edge, radius, num_basis=64, 
    atomref=None, task_mean=None, task_std=None, with_trace = True, trace_out_len=25, start_layer=0, use_w2v = True, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge, 
        irreps_node_embedding='64x0e+32x0o+32x1e+32x1o+16x2e+16x2o+16x3e+16x3o+16x4e+16x4o+16x5o+16x5e+8x6o+8x6e+4x7e', 
        irreps_edge_embedding='32x0e+16x1o+16x2e+16x3o+8x4e+8x5o+8x6e+4x7o', num_layers=4,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e+1x3o+1x4e+1x5o+1x6e+1x7o',
        max_radius=8.0,
        number_of_basis=64,
        irreps_feature='512x0e',
        irreps_head='64x0e+32x0o+32x1e+32x1o+16x2e+16x2o+16x3e+16x3o+16x4e+16x4o+16x5o+16x5e+8x6o+8x6e+4x7e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='64x0e+32x1o+32x1e+16x2e+16x2o+8x3o+8x3e+4x4e+4x4o+2x5e+2x5o', #暂时没用
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref, trace_out_len = trace_out_len
        )
    return model


class GaussianExpansion(torch.nn.Module):
    def __init__(self, min_val, max_val, num_basis):
        super(GaussianExpansion, self).__init__()
        self.centers = torch.linspace(min_val, max_val, num_basis)
        self.width = 0.5 * (max_val - min_val) / num_basis

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.exp(-((x - self.centers.to(x.device)) ** 2) / (2 * self.width ** 2))
