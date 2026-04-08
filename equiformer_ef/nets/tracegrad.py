import warnings
import os
import torch
import math
import random
from torch import nn
import torch.nn.functional as F
from e3nn.o3 import Irrep, Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct

from torch.autograd import grad

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
    
class equi_nonlin(nn.Module):
    def __init__(self, irreps_in):
        super(equi_nonlin, self).__init__()
        self.irreps_in = irreps_in

    def forward(self, x, retain_graph=True):

        ix = 0
        iw = 0
        out = []
        for mul, ir in self.irreps_in:
            field = x[:, ix: ix + mul * ir.dim]
            field = field.reshape(-1, mul, ir.dim)
            norm = field.norm(dim = -1)
            mask = norm < field.norm(dim = -1).mean(dim = -1).unsqueeze(-1)
            field[mask] = 0
            field = field.reshape(-1, mul * ir.dim)
            
            ix += mul * ir.dim
            iw += mul
            out.append(field)

        return torch.cat(out, dim=-1)
    
class equi_linear(nn.Module):
    def __init__(self, irreps_in, irreps_mid, irreps_out):
        super(equi_linear, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_mid = irreps_mid
        self.irreps_out = irreps_out
        self.net = torch.nn.ModuleList([
            Linear(irreps_in, irreps_mid),
            equi_nonlin(irreps_mid),
            Linear(irreps_mid, irreps_mid),
            equi_nonlin(irreps_mid),
            Linear(irreps_mid, irreps_out),
        ])
    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x