# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .registry import register_model

from e3nn import o3
from e3nn.o3 import Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode, script
from .tensor_product_rescale import LinearRS  # 自定义的 rescale 线性变换

from torch_scatter import scatter  # C++ 部署时需携带相应扩展库 .so/.dll（自定义算子）。:contentReference[oaicite:3]{index=3}

# 你自己的 EquiConv 从你工程里导入即可（不要重复贴）
from .model_jit import EquiConv

_MAX_ATOM_TYPE = 64 # Set to some large value

# Statistics of QM9 with cutoff radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 144 #18.03065905448718
_AVG_DEGREE = 15.57930850982666

# ------------------------
# utils
# ------------------------

def _segment_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    纯 scatter 实现的稀疏 softmax：对每个 group(index) 独立做 softmax。
    src: [E, ...], index: [E], 聚合维=0；支持广播到尾维。
    """
    # 减去各组最大值以稳定
    max_per_group = scatter(src, index, dim=0, dim_size=num_nodes, reduce='max')
    src_exp = torch.exp(src - max_per_group[index])
    denom = scatter(src_exp, index, dim=0, dim_size=num_nodes, reduce='sum')
    return src_exp / (denom[index] + 1e-12)  # 形状与 src 同，逐组归一化
# 行为与 torch_geometric.utils.softmax 一致（按 index 分组做 softmax）。:contentReference[oaicite:4]{index=4}


# ------------------------
# 可选：等变“非线性-梯度”模块
# ------------------------

@compile_mode('unsupported')  # 内部使用 autograd.grad；保留做实验/训练，但不纳入 TorchScript
class Equi_Nonlin_Grad_Module(nn.Module):
    def __init__(self, irreps_in, z_fea_len: int = 64, hidden_fea_len: int = 64):
        super().__init__()
        irr = o3.Irreps(irreps_in)
        self.fctp = FullyConnectedTensorProduct(irr, irr, f'{hidden_fea_len}x0e')
        self.nonlinear_layers = nn.Sequential(
            nn.Linear(hidden_fea_len, hidden_fea_len),
            nn.SiLU(),
            nn.Linear(hidden_fea_len, z_fea_len),
            nn.SiLU(),
            nn.Linear(z_fea_len, z_fea_len),
        )

    def forward(self, tensor_in, retain_graph: bool = True):
        # 警告：此路径不走 TorchScript，仅 Python 端可用
        from torch.autograd import grad
        x = self.fctp(tensor_in, tensor_in)
        x = self.nonlinear_layers(x)
        y = grad(outputs=x,
                 inputs=tensor_in,
                 grad_outputs=torch.ones_like(x),
                 retain_graph=retain_graph,
                 create_graph=retain_graph,
                 only_inputs=True,
                 allow_unused=True)[0]
        return x, y


# ------------------------
# Block：消息传递 + 注意力头
# ------------------------

@compile_mode('script')  # 含数据依赖分支，顶层 script
class Block(nn.Module):
    num_heads: int
    residual: bool

    def __init__(self,
                 irreps_node_in,
                 irreps_edge,
                 irreps_node_out,
                 irreps_sh,
                 num_heads: int,
                 edge_length_len: int,
                 tg_block: bool = False,
                 residual: bool = False):
        super().__init__()
        # 不把 Irreps 长期挂属性；仅用于构造子模块
        irr_node_in = Irreps(irreps_node_in)
        irr_node_out = Irreps(irreps_node_out)
        irr_sh = Irreps(irreps_sh)

        self.num_heads = int(num_heads)
        self.residual = bool(residual)

        # 预线性混合消息
        self.pre_lin = Linear(irr_node_in + irr_node_in + Irreps(f"{edge_length_len}x0e"), irr_node_in)

        # 两个 EquiConv：一个产生价值/权重（多头），一个可留作扩展
        self.tp = EquiConv(edge_length_len, irr_node_in, irr_sh, irr_node_in)
        self.tp2 = EquiConv(edge_length_len, irr_node_in, irr_sh, irr_node_in * self.num_heads)

        # 产生 attention 权重 alpha（不依赖图结构）
        self.lin_alpha = nn.Sequential(
            nn.Linear(edge_length_len, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.SiLU(),
            nn.Linear(64, self.num_heads),
        )

        # 聚合后映射到标量边特征（可作为后续 block 的附加输入）
        self.lin_edge = Linear(irr_node_in * self.num_heads, Irreps("64x0e"))
        self.lin_scalar = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.SiLU(),
            nn.Linear(64, 32),
        )

        # 节点更新映射
        if self.residual:
            self.lin = Linear(irr_node_in * (self.num_heads + 1), irr_node_out)
        else:
            self.lin = Linear(irr_node_in * (self.num_heads), irr_node_out)

    def forward(self,
                node_in: torch.Tensor,
                node_embed: torch.Tensor,
                edge_sh: torch.Tensor,
                edge_length_embedding: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                batch: torch.Tensor):
        # 消息：拼接 (src, dst, 边长嵌入) 再线性变换
        message = torch.cat((node_in[edge_src], node_in[edge_dst], edge_length_embedding), dim=-1)
        message = self.pre_lin(message)

        # 多头“值”
        value = self.tp2(message, edge_sh, edge_length_embedding, batch[edge_src])  # [E, H*C]
        value = value.reshape(value.shape[0], self.num_heads, -1)                    # [E, H, C]

        # 头内标量打分 → 稀疏 softmax（按 edge_dst 分组）
        alpha_logits = self.lin_alpha(edge_length_embedding)                         # [E, H]
        alpha = _segment_softmax(alpha_logits, edge_dst, num_nodes=node_in.size(0)).unsqueeze(-1)  # [E, H, 1]

        # 加权、合并
        edge_out = value * alpha                                                      # [E, H, C]
        edge_scalar = self.lin_edge(edge_out.reshape(edge_out.shape[0], -1))
        edge_scalar = self.lin_scalar(edge_scalar)

        # 汇聚到节点
        node_fea = scatter(edge_out, index=edge_dst, dim=0, dim_size=node_in.shape[0])  # [N, H, C]
        if self.residual:
            node_out = self.lin(torch.cat((node_fea.reshape(node_fea.shape[0], -1), node_in), dim=-1))
        else:
            node_out = self.lin(node_fea.reshape(node_fea.shape[0], -1))

        return node_out, edge_scalar


# ------------------------
# 其他小模块
# ------------------------

@compile_mode('trace')  # 仅张量计算，无数据分支
class GaussianExpansion(nn.Module):
    centers: torch.Tensor
    width: float

    def __init__(self, min_val: float, max_val: float, num_basis: int):
        super().__init__()
        # 用 register_buffer，随 .to(device) 迁移；不在 __init__ 里判 CUDA
        centers = torch.linspace(float(min_val), float(max_val), steps=int(num_basis))
        self.register_buffer('centers', centers)  # [B]
        self.width = 0.5 * (float(max_val) - float(min_val)) / float(num_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)                                  # [..., 1]
        return torch.exp(-((x - self.centers) ** 2) / (2.0 * (self.width ** 2)))



@compile_mode('script')
class NodeEmbeddingNetwork(nn.Module):
    max_atom_type: int
    embed_dim: int

    def __init__(self, irreps_node_embedding, max_atom_type: int = 64, embed_dim: int = 32, bias: bool = True):
        print("11111111111")
        """
        先用 nn.Embedding : [N] -> [N, embed_dim]
        再用 o3.Linear   : {embed_dim}x0e -> irreps_node_embedding
        """
        super().__init__()
        self.max_atom_type = int(max_atom_type)
        self.embed_dim = int(embed_dim)

        irr_emb = o3.Irreps(irreps_node_embedding)

        # 1) 常规嵌入：把类别 id 映射到稠密向量
        self.token_embed = nn.Embedding(self.max_atom_type, self.embed_dim)

        # 2) 等变线性：把 {embed_dim} 个标量通道映射到目标 irreps
        self.to_irreps = Linear(o3.Irreps(f"{self.embed_dim}x0e"), irr_emb)

        # 可选：轻微初始化（保持稳定即可，可按需调整）
        # nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        # 不再做之前基于 max_atom_type 的权重缩放；如需，可在此对 self.to_irreps.tp.weight 做等比例缩放。

    def forward(self, node_atom: torch.Tensor):
        """
        node_atom: LongTensor [N]
        返回：
          node_embedding  — 映射到 irreps 的特征，形如 [N, irr_emb.dim]
          atom_attr       — 这里返回稠密嵌入（[N, embed_dim]），可作为属性占位
          atom_dense      — 同上（为兼容原来三元返回）
        """
        # 常规嵌入
        dense = self.token_embed(node_atom)                 # [N, embed_dim], float32
        # 映射到目标 irreps
        node_embedding = self.to_irreps(dense)              # [N, irr_emb.dim]

        # 维持原函数的三元返回接口（下游基本未使用第二/第三项）
        atom_attr = dense
        atom_dense = dense
        return node_embedding, atom_attr, atom_dense


# ------------------------
# 顶层网络
# ------------------------

@compile_mode('script')  # 顶层含数据分支，用 script
class Nets(nn.Module):
    max_radius: float
    number_of_basis: int
    alpha_drop: float
    proj_drop: float
    out_drop: float
    drop_path_rate: float
    use_attn_head: bool
    num_heads: int
    num_layers: int
    edge_scalar_out: bool
    edge_scalar_out_len: int
    lmax: int

    def __init__(self,
                 irreps_in='64x0e',
                 irreps_node_embedding='128x0e+64x1o+32x2e',
                 irreps_edge_embedding='32x0e+16x1o+8x2e+4x3o+4x4e',
                 irreps_edge_output='1x0e',
                 num_layers=6,
                 irreps_node_attr='1x0e',
                 irreps_sh='1x0e+1x1e+1x2e',
                 max_radius=6.0,
                 number_of_basis=128,
                 fc_neurons=[64, 64],
                 irreps_feature='512x0e',
                 irreps_head='32x0e+16x1o+8x2e',
                 num_heads=4,
                 irreps_pre_attn=None,
                 rescale_degree=False,
                 nonlinear_message=False,
                 irreps_mlp_mid='128x0e+64x1e+32x2e',
                 use_attn_head=False,
                 norm_layer='layer',
                 alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
                 drop_path_rate=0.0,
                 mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        # 保存必要的标量/布尔属性
        self.max_radius = float(max_radius)
        self.number_of_basis = int(number_of_basis)
        self.alpha_drop = float(alpha_drop)
        self.proj_drop = float(proj_drop)
        self.out_drop = float(out_drop)
        self.drop_path_rate = float(drop_path_rate)
        self.use_attn_head = bool(use_attn_head)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)

        # 这些 Irreps 只在构造子模块时使用；不长期保存
        irr_node_emb = o3.Irreps(irreps_node_embedding)
        self.lmax = int(irr_node_emb.lmax)  # 仅保存整数 lmax
        irr_feature = o3.Irreps(irreps_feature)
        irr_head = o3.Irreps(irreps_head)
        irr_sh = o3.Irreps(irreps_sh)
        irr_edge_mid = o3.Irreps(irreps_edge_embedding)

        # 嵌入与基函数
        self.atom_embed = NodeEmbeddingNetwork(irr_node_emb, _MAX_ATOM_TYPE)
        self.rbf = GaussianExpansion(0.0, 10.0, num_basis=self.number_of_basis)

        # SH 用“模块”版本，JIT 友好
        self.sh = SphericalHarmonics(irr_sh, normalize=True, normalization='component')

        # 边特征预处理
        self.edge_prelin = Linear(irr_sh, irr_edge_mid)

        # 头部 MLP（能量回归）
        self.head = nn.Sequential(
            nn.Linear(irr_feature.dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # 单原子能量基线
        self.e0_lin = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Block 堆叠
        self.blocks = nn.ModuleList()
        self.edge_scalar_out = False
        self.edge_scalar_out_len = 32 if self.edge_scalar_out else 0
        residual = True

        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                blk = Block(irr_node_emb, irr_edge_mid, irr_node_emb, irr_sh,
                            self.num_heads, edge_length_len=self.number_of_basis + self.edge_scalar_out_len,
                            tg_block=False, residual=residual)
            else:
                blk = Block(irr_node_emb, irr_edge_mid, irr_feature, irr_sh,
                            self.num_heads, edge_length_len=self.number_of_basis + self.edge_scalar_out_len,
                            tg_block=False, residual=residual)
            self.blocks.append(blk)

        # 常数原子能（若不需要可删掉或改为参数化）
        atom_energy = torch.tensor([
            -1526.13972398047, -441.1224924322264, -1367.358861870081, -1102.451062360772,
            -58.86980064545264, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], dtype=torch.float32)
        self.register_buffer('atom_energy', atom_energy, persistent=True)

        # 任务相关统计
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        # self.register_buffer('atomref', atomref)

    def _energy_pass(self,
                     node_atom: torch.Tensor,
                     edge_src: torch.Tensor,
                     edge_dst: torch.Tensor,
                     edge_vec: torch.Tensor,
                     batch: torch.Tensor,
                     R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        仅做前向能量与中间量的计算（Script 友好），供 forward 与力/应力计算复用。
        返回：
          node_features, e0, edge_scalar, edge_sh
        """
        # 节点嵌入
        node_features, atom_attr, atom_onehot = self.atom_embed(node_atom)  # node_features: [N, emb_dim]

        # e0（可学习的每原子基线）
        e0 = self.e0_lin(node_features[:, :64])  # [N,1]

        # 边长及其 RBF
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)  # [E, B]

        # Spherical Harmonics：用模块，R 为 3x3；l=1 时 D(R)=R，直接矢量右乘 R 即可
        x_rot = edge_vec[:, [1, 2, 0]] @ R  # 与你原始写法保持一致的坐标顺序与旋转
        edge_sh = self.sh(x_rot)            # [E, sum mul*dim]

        # 边不变特征
        edge_features = self.edge_prelin(edge_sh)

        # 可能拼接来自上层 block 的边标量
        edge_scalar = edge_length_embedding
        if self.edge_scalar_out:
            edge_scalar = torch.cat((edge_length_embedding,
                                     torch.zeros_like(edge_length_embedding[:, :self.edge_scalar_out_len])),
                                    dim=-1)

        # 逐层消息传递
        node_embed = node_features
        for blk_idx, blk in enumerate(self.blocks):
            node_features, edge_scalar_out = blk(node_in=node_features, node_embed=node_embed,
                                                 edge_sh=edge_sh, edge_length_embedding=edge_scalar,
                                                 edge_src=edge_src, edge_dst=edge_dst, batch=batch)
            if self.edge_scalar_out:
                edge_scalar = torch.cat((edge_length_embedding, edge_scalar_out), dim=-1)

        return node_features, e0, edge_scalar, edge_sh

    def forward(self,
                node_atom: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                edge_vec: torch.Tensor,
                batch: torch.Tensor,
                R: Optional[torch.Tensor] = None,
                ):
        """
        Script 模式：返回 (energy, forces, virial)
        - energy: [B]   （按 batch 汇聚）
        - forces: [N,3] （在脚本化时为 0 占位；在 Python 端训练时会计算真实梯度）
        - virial: [B,3,3] （脚本化时 0，占位）
        """
        if R is None:
            R = torch.eye(3, dtype=edge_vec.dtype, device=edge_vec.device)

        node_features, e0, edge_scalar, edge_sh = self._energy_pass(
            node_atom, edge_src, edge_dst, edge_vec, batch, R
        )

        # 回归能量（节点 → 标量/每原子）
        energy_per_atom = self.head(node_features)[:, 0] + e0[:, 0] - 150.0
        energy = scatter(energy_per_atom, batch, dim=0)  # [B]

        # ----------------
        # 力/应力：TorchScript 下保留接口但返回 0（占位）
        # ----------------
        if torch.jit.is_scripting():
            forces = torch.zeros(node_atom.shape[0], 3, device=edge_vec.device, dtype=edge_vec.dtype)
            virial1 = torch.zeros(int(batch.max().item()) + 1, 3, 3, device=edge_vec.device, dtype=edge_vec.dtype)
            return energy, forces, virial1

        # Python / 训练 时：用 autograd 计算真实力与应力
        edge_vec_req = edge_vec.requires_grad_(True)
        node_features2, e02, edge_scalar2, edge_sh2 = self._energy_pass(
            node_atom, edge_src, edge_dst, edge_vec_req, batch, R
        )
        energy_per_atom2 = self.head(node_features2)[:, 0] + e02[:, 0] - 150.0
        energy2 = scatter(energy_per_atom2, batch, dim=0)
        force_edge = torch.autograd.grad(energy2.sum(), edge_vec_req, create_graph=True, only_inputs=True,
                                         allow_unused=True)[0]  # [E,3]

        mask = edge_vec_req.norm(dim=-1) != 0
        forces = torch.zeros(node_atom.shape[0], 3, device=edge_vec.device, dtype=edge_vec.dtype)
        filtered_edge_index_0 = edge_src[mask]
        filtered_edge_index_1 = edge_dst[mask]
        filtered_force_edge = force_edge[mask]
        forces.index_add_(0, filtered_edge_index_1, -filtered_force_edge)
        forces.index_add_(0, filtered_edge_index_0, filtered_force_edge)

        virial = -edge_vec_req.unsqueeze(2) * force_edge.unsqueeze(1)  # [E,3,3]
        virial1 = scatter(virial, batch[edge_src], dim=0)              # [B,3,3]

        return energy, forces, virial1


# ------------------------
# 工厂函数（保持原接口）
# ------------------------

@register_model
def f_model(irreps_in, irreps_edge, radius, num_basis=128,
            atomref=None, task_mean=None, task_std=None,
            with_trace=True, trace_out_len=25, start_layer=0, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge,
        irreps_node_embedding='64x0e+32x1o+16x2e',
        irreps_edge_embedding='64x0e+32x1o+16x2e',
        num_layers=3,
        irreps_node_attr='1x0e',
        irreps_sh='1x0e+1x1o+1x2e',
        max_radius=8.0,
        number_of_basis=64, fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='64x0e+32x1o+16x2e',
        num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='64x0e+32x1o+16x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
    )
    return model


@register_model
def f_model_large(irreps_in, irreps_edge, radius, num_basis=128,
                  atomref=None, task_mean=None, task_std=None,
                  with_trace=True, trace_out_len=25, start_layer=0, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge,
        irreps_node_embedding='64x0e+32x1o+32x1e+16x2e',
        irreps_edge_embedding='64x0e+32x1o+32x1e+16x2e',
        num_layers=3,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
        max_radius=8.0,
        number_of_basis=64, fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='64x0e+32x1o+32x1e+16x2e',
        num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='64x0e+32x1o+32x1e+16x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
    )
    return model


@register_model
def my_model_complex(irreps_in, irreps_edge, radius, num_basis=128,
                     atomref=None, task_mean=None, task_std=None,
                     with_trace=True, trace_out_len=25, start_layer=0, **kwargs):
    model = Nets(
        irreps_in=irreps_in, irreps_edge_output=irreps_edge,
        irreps_node_embedding='32x0e+16x1o+8x2e+8x3o+8x4e',
        irreps_edge_embedding='32x0e+16x1o+8x2e+8x3o+8x4e',
        num_layers=4,
        irreps_node_attr='1x0e',
        irreps_sh='1x0e+1x1o+1x2e+1x3o+1x4e',
        max_radius=8.0,
        number_of_basis=64, fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e+8x3o+8x4e',
        num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='32x0e+16x1o+8x2e+8x3o+8x4e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
    )
    return model
