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

class Block(torch.nn.Module):
    def __init__(self, irreps_node_in, irreps_edge, irreps_node_out, irreps_sh, num_heads, edge_length_len, tg_block = False, residual = False):
        super().__init__()
        self.irreps_node_in = Irreps(irreps_node_in)
        self.irreps_edge = Irreps(irreps_edge)
        self.irreps_node_out = Irreps(irreps_node_out)
        self.irreps_sh = Irreps(irreps_sh)
        self.num_heads = num_heads
        self.pre_lin = Linear(self.irreps_node_in+self.irreps_node_in + Irreps(f"{edge_length_len}x0e"), self.irreps_node_in)
        self.tp = EquiConv(edge_length_len, self.irreps_node_in, self.irreps_sh, self.irreps_node_in)
        self.tp2 = EquiConv(edge_length_len, self.irreps_node_in, self.irreps_sh, self.irreps_node_in * self.num_heads)
        self.lin_alpha = torch.nn.Sequential(
            torch.nn.Linear(edge_length_len, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.num_heads)
            )
        # if tg_block:
        #     self.tg_block = Equi_Nonlin_Grad_Module(self.irreps_node_out, z_fea_len = 49)
        # else:
        #     self.tg_block = None
        # self.lin_test = Linear(self.irreps_node_in + self.irreps_sh, self.irreps_node_in * self.num_heads)
        self.residual = residual
        if residual:
            self.lin = Linear(self.irreps_node_in * (self.num_heads+1), self.irreps_node_out)
        else:
            self.lin = Linear(self.irreps_node_in * (self.num_heads), self.irreps_node_out)
        
        self.lin_edge = Linear(self.irreps_node_in * self.num_heads, Irreps("64x0e"))
        self.lin_scalar = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32)
            )
        # self.norm1 = get_norm_layer('layer')(self.irreps_node_in)
        # self.norm2 = get_norm_layer('layer')(self.irreps_node_out)
        # # self.lin_fnn1 = Linear(self.irreps_node_out, self.irreps_node_out)
        # self.fnn = EquivariantNonlin(irreps_node_out)
        # mul_alpha = get_mul_0(self.irreps_node_in * self.num_heads)
        # self.irreps_alpha = Irreps(f"{mul_alpha}x0e")
        # mul_alpha_head = mul_alpha // num_heads
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        # self.sep_alpha = Linear(self.irreps_node_in * self.num_heads, self.irreps_alpha)
        # self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
        #     [SmoothLeakyReLU(0.2)])

    def forward(self, node_in, node_embed, edge_sh, edge_length_embedding, edge_src, edge_dst, batch):
        # node_fea = self.norm1(node_in)
        message = torch.cat((node_in[edge_src], node_in[edge_dst], edge_length_embedding), dim = -1)
        message = self.pre_lin(message)
        value = self.tp2(message, edge_sh, edge_length_embedding,
                        batch[edge_src])
        # value = self.lin_test(torch.cat((message, edge_sh), dim = -1))
        # value = self.tp2(value, edge_sh, None, batch[edge_src])
        
        # alpha = self.sep_alpha(value).reshape(value.shape[0], self.num_heads, -1)
        # alpha = self.alpha_act(alpha)
        # alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = self.lin_alpha(edge_length_embedding)
        # alpha = torch.ones_like(alpha)
        value = value.reshape(value.shape[0], self.num_heads, -1)
        edge_out = value * alpha.unsqueeze(-1)
        edge_scalar = self.lin_edge(edge_out.reshape(edge_out.shape[0], -1))
        edge_scalar = self.lin_scalar(edge_scalar)

        alpha = torch_geometric.utils.softmax(alpha, edge_dst).unsqueeze(-1)
        edge_fea = value * alpha
        
        node_fea = scatter(edge_fea, index=edge_dst, dim=0, dim_size=node_in.shape[0])
        
        if self.residual:
            node_out = self.lin(torch.cat((node_fea.reshape(node_fea.shape[0], -1), node_in), dim = -1))
        else:
            node_out = self.lin(node_fea.reshape(node_fea.shape[0], -1))
        # if self.tg_block is not None:
        #     node_out = self.tg_block(node_out)[1] + node_out

        # node_out = self.norm2(node_out)
        # node_fea = self.fnn(node_out)
        # node_out = node_out + node_fea

        return node_out, edge_scalar





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
        
        self.rbf = GaussianExpansion(0, 10, num_basis=self.number_of_basis)
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
                              edge_length_len = self.number_of_basis + self.edge_scalar_out_len, 
                              tg_block = False, residual = residual)
            else:
                block = Block(self.irreps_node_embedding, self.irreps_edge_mid, 
                              self.irreps_feature, self.irreps_sh, num_heads,
                              edge_length_len = self.number_of_basis + self.edge_scalar_out_len, 
                              residual=residual)

            self.blocks.append(block)


    @torch.enable_grad()
    def forward(self, node_atom, edge_src, edge_dst, edge_vec, batch, R = torch.eye(3), **kwargs):
        #pos = pos.requires_grad_(True)
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

        for blk_idx, blk in enumerate(self.blocks):
            node_features, edge_scalar_out = blk(node_in = node_features, node_embed = node_embed, edge_sh = edge_sh, 
                                edge_length_embedding = edge_scalar,  
                                edge_src = edge_src, edge_dst = edge_dst, batch = batch)
            if self.edge_scalar_out:
                edge_scalar = torch.cat((edge_length_embedding, edge_scalar_out), dim = -1)

        

        # energy = self.head(node_features)[:, 0] + e0[:, 0] + atom_energy[node_atom]
        base_energy = self.base_energy.to(device=node_atom.device)
        energy = self.head(node_features)[:, 0] + e0[:, 0] + base_energy[node_atom]
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
