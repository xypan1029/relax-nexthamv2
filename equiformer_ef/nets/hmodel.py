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
            torch.nn.Dropout(p=0.1),
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
        node_msg = scatter(value * alpha, index=edge_dst, dim=0, dim_size=node_fea.shape[0])

        node_msg = self.heads2vec(node_msg)
        # edge_msg = self.heads2vec_edge(edge_msg)

        node_out = self.lin(node_msg)
        # edge_out = self.lin_edge(edge_msg)

        node_out = node_out + self.node_skip(node_fea_res)
        # edge_out = edge_out + self.edge_skip(edge_fea_res)

        # ---------------------------------
        # FFN placeholder
        #   node_out = node_out + FFN(norm(node_out))
        # ---------------------------------
        # TODO: add FFN here

        return node_out, None




class Nets(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding='128x0e+64x1o+32x2e',
        irreps_edge_embedding='32x0e+16x1o+8x2e+4x3o+4x4e',
irreps_edge_output='1x0e+1x0e+1x1o+1x1o+1x2e+1x0e+1x0e+1x1o+1x1o+1x2e+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x1o+1x1o+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+1x2e+1x2e+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x0e+1x1e+1x2e+1x3e+1x4e',
        num_layers=6,
        irreps_sh='1x0e+1x1e+1x2e',
        number_of_basis=128,
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e',
        num_heads=4,
        norm_layer='layer',
        atomref=None,
    ):
        super().__init__()

        self.register_buffer('atomref', atomref)

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_edge_attr = o3.Irreps(irreps_sh)
        self.irreps_head = o3.Irreps(irreps_head)
        self.base_energy = None
        self.task_mean = 0
        self.task_std = 1

        self.irreps_edge_mid = o3.Irreps(irreps_edge_embedding)
        self.irreps_edge_output = o3.Irreps(irreps_edge_output)

        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.rbf = GaussianExpansion(0, 10, num_basis=number_of_basis)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1)
        )

        self.e0_lin = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1)
        )

        self.blocks = torch.nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                block = Block(
                    self.irreps_node_embedding,
                    self.irreps_head,
                    Irreps(f"{number_of_basis}x0e"),
                    self.irreps_node_embedding,
                    self.irreps_node_embedding,
                    self.irreps_sh,
                    num_heads
                )
            else:
                block = Block(
                    self.irreps_node_embedding,
                    Irreps("128x0e"),
                    Irreps(f"{number_of_basis}x0e"),
                    self.irreps_edge_output.sort()[0].simplify(),
                    self.irreps_feature,
                    self.irreps_sh,
                    num_heads
                )
            self.blocks.append(block)

    @torch.enable_grad()
    def forward(self, edge_src, edge_dst, edge_vec, batch, node_atom):
        edge_vec = edge_vec.requires_grad_(True)

        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec[:, [1, 2, 0]],
            normalize=True,
            normalization='component'
        )

        node_features, atom_attr, atom_onehot = self.atom_embed(node_atom)
        e0 = self.e0_lin(node_features[:, :64])

        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)

        D_dict = compute_D(edge_vec, l_max=7)
        for blk in self.blocks:
            node_features, _ = blk(
                node_fea=node_features,
                edge_fea=edge_length_embedding,
                edge_sh=edge_sh,
                edge_length_embedding=edge_length_embedding,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_vec=edge_vec,
                D_dict=D_dict,
                batch=batch
            )

        energy = self.head(node_features)[:, 0] + e0[:, 0] + self.base_energy[node_atom]
        energy = scatter(energy, batch, dim=0)

        mask = edge_vec.norm(dim=-1) != 0
        force_edge = torch.autograd.grad(
            energy.sum(),
            edge_vec,
            create_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]

        forces = torch.zeros(
            node_atom.shape[0], 3,
            device=edge_vec.device,
            dtype=edge_vec.dtype
        )
        filtered_edge_index_0 = edge_src[mask]
        filtered_edge_index_1 = edge_dst[mask]
        filtered_force_edge = force_edge

        forces.index_add_(0, filtered_edge_index_1, -filtered_force_edge)
        forces.index_add_(0, filtered_edge_index_0, filtered_force_edge)

        virial = -edge_vec.unsqueeze(2) * force_edge.unsqueeze(1)
        virial1 = scatter(virial, batch[edge_src], dim=0)

        return energy, forces, virial1

@register_model
def h_model(irreps_in, irreps_edge, radius, num_basis=64, 
    atomref=None, task_mean=None, task_std=None, with_trace = True, trace_out_len=25, start_layer=0, use_w2v = True, **kwargs):
    model = Nets(
        irreps_edge_output='64x0e',
        irreps_node_embedding='64x0e+32x1o+16x2e+16x3o+16x4e',
        irreps_edge_embedding='32x0e+16x1o+16x2e+16x3o+8x4e+8x5o+8x6e+4x7o',
        num_layers=4,
        irreps_sh='1x0e+1x1o+1x2e+1x3o+1x4e+1x5o+1x6e+1x7o',
        number_of_basis=64,
        irreps_feature='512x0e',
        irreps_head='64x0e+32x1o+16x2e+16x3o+16x4e',
        num_heads=4)
    return model


class GaussianExpansion(torch.nn.Module):
    def __init__(self, min_val, max_val, num_basis):
        super(GaussianExpansion, self).__init__()
        self.centers = torch.linspace(min_val, max_val, num_basis)
        self.width = 0.5 * (max_val - min_val) / num_basis

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.exp(-((x - self.centers.to(x.device)) ** 2) / (2 * self.width ** 2))
