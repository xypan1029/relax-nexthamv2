import warnings
import os
import torch
import math
import random
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from e3nn import o3


class MergeFeaturesNet(nn.Module):
    def __init__(self, input_sizes, hidden_size, output_size):
        super(MergeFeaturesNet, self).__init__()
        self.fc1 = nn.Linear(input_sizes[0], hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(input_sizes[1], hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(input_sizes[2], hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.merge_fc = nn.Linear(hidden_size * 3, output_size)
        self.merge_ln = nn.LayerNorm(output_size)

    def forward(self, x1, x2, x3):
        h1 = F.silu(self.ln1(self.fc1(x1)))
        h2 = F.silu(self.ln2(self.fc2(x2)))
        h3 = F.silu(self.ln3(self.fc3(x3)))
        merged = torch.cat((h1, h2, h3), dim=1) 
        output = F.silu(self.merge_ln(self.merge_fc(merged)))
        return output


class GaussianExpansion(torch.nn.Module):
    def __init__(self, min_val, max_val, num_basis):
        super(GaussianExpansion, self).__init__()
        self.centers = torch.linspace(min_val, max_val, num_basis).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.width = 0.5 * (max_val - min_val) / num_basis

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.exp(-((x - self.centers) ** 2) / (2 * self.width ** 2))

class Graph_Transformer_Block(torch.nn.Module):
    def __init__(self, dim_key, dim_node, dim_output_node, dim_edge, dim_output_edge, n_head=4):
        super(Graph_Transformer_Block, self).__init__()
        self.n_head = n_head
        dim_divide = dim_output_node//n_head
        dim_divide_recover = dim_divide*n_head
        h_q_list = []
        for _ in range(n_head):
            h_q_list.append(torch.nn.Linear(dim_node+dim_edge+dim_node, dim_key))
        self.h_q = torch.nn.ModuleList(h_q_list)
        h_k_list = []
        for _ in range(n_head):
            h_k_list.append(torch.nn.Linear(dim_node+dim_edge+dim_node, dim_key))
        self.h_k = torch.nn.ModuleList(h_k_list)
        h_v_list = []
        for _ in range(n_head):
            h_v_list.append(torch.nn.Linear(dim_node+dim_edge+dim_node, dim_divide))    
        self.h_v = torch.nn.ModuleList(h_v_list)
        self.v_linear = torch.nn.Linear(dim_divide_recover, dim_output_node)
        self.residual_convert = torch.nn.Linear(dim_node, dim_output_node)
        self.w_1 = torch.nn.Linear(dim_output_node, dim_output_node) # position-wise
        self.w_2 = torch.nn.Linear(dim_output_node, dim_output_node) # position-wise
        self.act_n = torch.nn.SiLU(inplace=False)
        self.w_3 = torch.nn.Linear(2*dim_output_node+dim_edge, dim_output_edge)
        self.w_4 = torch.nn.Linear(dim_output_edge, dim_output_edge)
        self.norm_a = torch.nn.LayerNorm(dim_output_node, eps=1e-6)
        self.norm_ffn = torch.nn.LayerNorm(dim_output_node, eps=1e-6)
        self.norm_e = torch.nn.LayerNorm(dim_output_edge, eps=1e-6)
        self.act_e = torch.nn.SiLU(inplace=False)


    def MultiHeadAttention(self, f_q, f_k, f_v, node_fea_in, edge_dst):
        residual = node_fea_in
        v_list = []
        for head in range(self.n_head):
            h_q_head = self.h_q[head](f_q)
            h_k_head = self.h_k[head](f_k)
            dot_res = torch.sum(h_q_head.mul(h_k_head), dim=-1, keepdim = True)
            dot_res = dot_res/(math.sqrt(f_v.shape[-1]))
            dot_res = torch.clip(dot_res, min=-5.0, max=5.0)
            exp = dot_res.exp()
            z = scatter(exp, edge_dst, dim=0, dim_size=len(node_fea_in))
            z[z == 0] = 1.0
            alpha = exp / z[edge_dst]
            v_list.append(scatter(alpha*self.h_v[head](f_v), edge_dst, dim=0, dim_size=len(node_fea_in)))
        v = torch.cat(v_list, dim = -1)
        v = self.v_linear(v)
        if int(residual.shape[-1]) != int(v.shape[-1]):
            residual = self.residual_convert(residual)
        v = residual+v
        v = self.norm_a(v)
        return v

    def PositionwiseFeedForward(self, f_v):
        residual = f_v
        v = self.w_1(f_v)
        v += residual
        v = self.norm_ffn(v)
        v = self.act_n(v)
        v = self.w_2(v)
        return v

    def forward(self, node_fea_in, edge_fea_in, edge_src, edge_dst):
        q = torch.cat((node_fea_in[edge_src], edge_fea_in, node_fea_in[edge_dst]), dim = -1)
        k = torch.cat((node_fea_in[edge_src], edge_fea_in, node_fea_in[edge_dst]), dim = -1)
        v = torch.cat((node_fea_in[edge_src], edge_fea_in, node_fea_in[edge_dst]), dim = -1)
        v = self.MultiHeadAttention(q, k, v, node_fea_in, edge_dst)
        v = self.PositionwiseFeedForward(v)
        new_node_fea = v
        new_edge_fea = torch.cat((new_node_fea[edge_src], edge_fea_in, new_node_fea[edge_dst]), dim=-1)
        new_edge_fea = self.w_3(new_edge_fea)
        new_edge_fea = self.norm_e(new_edge_fea)
        new_edge_fea = self.act_e(new_edge_fea)
        new_edge_fea = self.w_4(new_edge_fea)
        return new_node_fea, new_edge_fea

class Hamiltonian_Transformer(torch.nn.Module):
    def __init__(self, input_len, output_len, radius, num_basis, trace_out_len, block_num=20, hidden_fea_len=256, n_head=8, eage_sh_irreps='1x0e+1x1o+1x2e+1x3o+1x4e+1x5o+1x6e+1x7o'):
        super(Hamiltonian_Transformer, self).__init__()
        self.eage_sh_irreps = eage_sh_irreps
        self.rbf = GaussianExpansion(0, radius, num_basis=num_basis)
        self.output_len = output_len
        self.task_mean = 0.0
        self.task_std = 1.0
        merge_list = []
        block_list = []
        for b_idx in range(block_num):
            if b_idx < block_num-1:
                if b_idx==0:
                    merge_list.append(MergeFeaturesNet([input_len, o3.Irreps(self.eage_sh_irreps).dim, num_basis], hidden_fea_len, hidden_fea_len))
                else:
                    merge_list.append(MergeFeaturesNet([hidden_fea_len, o3.Irreps(self.eage_sh_irreps).dim, num_basis], hidden_fea_len, hidden_fea_len))
                block_list.append(Graph_Transformer_Block(hidden_fea_len, hidden_fea_len, hidden_fea_len, hidden_fea_len, hidden_fea_len))
            else:
                merge_list.append(MergeFeaturesNet([hidden_fea_len, o3.Irreps(self.eage_sh_irreps).dim, num_basis], hidden_fea_len, hidden_fea_len))
                block_list.append(Graph_Transformer_Block(hidden_fea_len, hidden_fea_len, hidden_fea_len, hidden_fea_len, output_len+trace_out_len))
        self.merge_list = torch.nn.ModuleList(merge_list)
        self.block_list = torch.nn.ModuleList(block_list)
    
    def forward(self, weak_ham, node_num, edge_vec, edge_src, edge_dst):
        edge_sh = o3.spherical_harmonics(l=self.eage_sh_irreps, x=edge_vec[:, [1,2,0]], normalize=True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        mask = edge_src == edge_dst
        edge_fea = weak_ham
        for b_idx in range(len(self.block_list)):
            # print(b_idx)
            edge_fea = self.merge_list[b_idx](edge_fea, edge_sh, edge_length_embedding)
            if b_idx == 0:
                filtered_edge_indices = edge_src[mask]
                filtered_edge_features = edge_fea[mask]
                node_fea = scatter(filtered_edge_features, filtered_edge_indices, dim=0, dim_size=node_num)
            node_fea, edge_fea = self.block_list[b_idx](node_fea, edge_fea, edge_src, edge_dst)
        return edge_fea[:, :self.output_len], edge_fea[:, self.output_len:]
        


