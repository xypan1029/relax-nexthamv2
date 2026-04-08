import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

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
from .tracegrad import Equi_Nonlin_Grad_Module


_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 128 # Set to some large value

# Statistics of QM9 with cutoff radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 144 #18.03065905448718
_AVG_DEGREE = 15.57930850982666


class Block(torch.nn.Module):
    def __init__(self, irreps_node_in, irreps_edge, irreps_node_out, irreps_sh, num_heads):
        super().__init__()
        self.irreps_node_in = Irreps(irreps_node_in)
        self.irreps_edge = Irreps(irreps_edge)
        self.irreps_node_out = Irreps(irreps_node_out)
        self.irreps_sh = Irreps(irreps_sh)
        self.num_heads = num_heads
        self.pre_lin = Linear(self.irreps_node_in+self.irreps_node_in + Irreps("64x0e"), self.irreps_node_in)
        self.tp = EquiConv(64, self.irreps_node_in, self.irreps_sh, self.irreps_node_in)
        self.tp2 = EquiConv(64, self.irreps_node_in, self.irreps_sh, self.irreps_node_in * self.num_heads)
        self.lin_alpha = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            # torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.num_heads)
            )
        # self.tg_block = Equi_Nonlin_Grad_Module(self.irreps_node_in, z_fea_len = 64)
        self.lin = Linear(self.irreps_node_in * self.num_heads, self.irreps_node_out)

    def forward(self, node_fea, edge_sh, edge_length_embedding, edge_src, edge_dst, batch):
        message = torch.cat((node_fea[edge_src], node_fea[edge_dst], edge_length_embedding), dim = -1)
        message = self.pre_lin(message)
        value = self.tp2(message, edge_sh, edge_length_embedding, batch[edge_src])
        # value = self.tp2(value, edge_sh, None, batch[edge_src])
        value = value.reshape(value.shape[0], self.num_heads, -1)
        alpha = self.lin_alpha(edge_length_embedding)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst).unsqueeze(-1)
        edge_fea = value * alpha
        #edge_fea = self.tg_block(edge_fea)[1] + edge_fea
        node_fea = scatter(edge_fea, index=edge_dst, dim=0, dim_size=node_fea.shape[0])

        node_out = self.lin(node_fea.reshape(node_fea.shape[0], -1))
        return node_out





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
        
        self.rbf = GaussianExpansion(0, 10, num_basis=64)
        
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

        for i in range(num_layers):
            if i < num_layers - 1:
                block = Block(self.irreps_node_embedding, self.irreps_edge_mid, 
                              self.irreps_node_embedding, self.irreps_sh, num_heads)
            else:
                block = Block(self.irreps_node_embedding, self.irreps_edge_mid, 
                              self.irreps_feature, self.irreps_sh, num_heads)
            self.blocks.append(block)


    # @torch.enable_grad()
    def forward(self, node_atom, edge_src, edge_dst, edge_vec, batch, R = torch.eye(3), **kwargs):
        #pos = pos.requires_grad_(True)
        # edge_vec = edge_vec.requires_grad_(True)
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec[:, [1,2,0]] @ o3.Irrep(1, 1).D_from_matrix(R).to(edge_vec.device), normalize=True, normalization='component')
        node_features, atom_attr, atom_onehot = self.atom_embed(node_atom)

        e0 = self.e0_lin(node_features[:, :64])
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)

        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        edge_features = self.edge_prelin(edge_sh)
        edge_features = torch.cat((edge_length_embedding, torch.zeros_like(edge_features[:, self.number_of_basis:])), dim = -1)
        
        edge_features_invar_list = []
        for blk_idx, blk in enumerate(self.blocks):
            node_features = blk(node_fea = node_features, edge_sh = edge_sh, 
                                edge_length_embedding = edge_length_embedding, 
                                edge_src = edge_src, edge_dst = edge_dst, batch = batch)
        
        return None, None, node_features
        atom_energy = torch.tensor([
            -1526.13972398047,
            -441.1224924322264,
            -1367.358861870081,
            -1102.451062360772,
            -58.86980064545264,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], device = e0.device)
        

        # energy = self.head(node_features)[:, 0] + e0[:, 0] + atom_energy[node_atom]
        energy = self.head(node_features)[:, 0] + e0[:, 0] - 150

        energy = scatter(energy, batch, dim=0)

        mask = edge_vec.norm(dim = -1) != 0
        force_edge = torch.autograd.grad(energy.sum(), edge_vec, create_graph=True, only_inputs=True, allow_unused=True)[0][mask]
        forces = torch.zeros(node_atom.shape[0], 3, device = edge_vec.device, dtype = edge_vec.dtype)
        filtered_edge_index_0 = edge_src[mask]
        filtered_edge_index_1 = edge_dst[mask]
        filtered_force_edge = force_edge
        forces.index_add_(0, filtered_edge_index_1, -filtered_force_edge)
        forces.index_add_(0, filtered_edge_index_0, filtered_force_edge)
        virial = -edge_vec.unsqueeze(2) * edge_vec.unsqueeze(1)
        virial = scatter(virial, batch[edge_src], dim = 0)

        return energy, forces, Z

@register_model
def z_model(irreps_in, irreps_edge, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, with_trace = True, trace_out_len=25, start_layer=0, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge, 
        irreps_node_embedding='32x0e+8x1o+8x1e+4x2e', 
        irreps_edge_embedding='32x0e+8x1o+8x1e+4x2e', num_layers=3,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=8.0,
        number_of_basis=64, fc_neurons=[64, 64],
        irreps_feature='1x0e+1x1e+1x2e',
        irreps_head='16x0e+8x1o+8x1e+4x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='32x0e+8x1o+8x1e+4x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
        )
    return model

@register_model
def my_model_large(irreps_in, irreps_edge, radius, num_basis=128, 
    atomref=None, task_mean=None, task_std=None, with_trace = True, trace_out_len=25, start_layer=0, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge, 
        irreps_node_embedding='128x0e+64x1o+32x2e', 
        irreps_edge_embedding='128x0e+64x1o+32x2e', num_layers=4,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=8.0,
        number_of_basis=64, fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='128x0e+64x1o+32x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='128x0e+64x1o+32x2e',
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
        self.centers = torch.linspace(min_val, max_val, num_basis).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.width = 0.5 * (max_val - min_val) / num_basis

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.exp(-((x - self.centers) ** 2) / (2 * self.width ** 2))
