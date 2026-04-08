# test_equivariance_so2_conv.py
# -*- coding: utf-8 -*-
import torch
from e3nn import o3
import math

# === 把这行改成你的实际导入 ===
from SO2_tools import SO2_Convolution
# ============================

def random_so3(device, dtype):
    """用 QR + 行列式修正产生 SO(3) 随机旋转矩阵。"""
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]
    return Q  # [3,3]

@torch.no_grad()
def check_equivariance_SO2Conv():
    torch.manual_seed(0)

    # -------- 配置 --------
    E = 5
    fea_weight_dim = 64
    irreps_in  = "3x0e + 3x1o + 4x2e+ 5x3o"
    irreps_out = "3x0e + 3x1o + 4x2e+ 5x3o"

    dim_in  = o3.Irreps(irreps_in).dim
    dim_out = o3.Irreps(irreps_out).dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dtype  = torch.float32

    # -------- 随机输入 --------
    fea_edge   = torch.randn(E, dim_in,  device=device, dtype=dtype)
    edge_vec   = torch.randn(E, 3,       device=device, dtype=dtype)
    fea_weight = torch.randn(E, fea_weight_dim, device=device, dtype=dtype)
    fea_edge[0, 0], fea_edge[0, 1], fea_edge[0, 2] = 1, 0, 0
    edge_vec[0, 0], edge_vec[0, 1], edge_vec[0, 2] = 0, 0, 1

    # -------- 构建模型 --------
    model = SO2_Convolution(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        fea_weight_dim=fea_weight_dim,
        gate_dim=16,         # 随便给个 gate 维度（只参与网络内部，等变性只检查 edge_out）
        mlp_hidden=64
    ).to(device)

    # baseline 输出
    # edge_vec = edge_vec[:, [1,2,0]]
    from SO3_tools import compute_D
    D_dict = compute_D(edge_vec, l_max=7)
    out, gate = model(fea_edge, edge_vec, D_dict, fea_weight)     # out: [E, dim_out]

    # -------- 生成旋转，并旋转输入 --------
    
    R = -random_so3(device, dtype)    
    # R = -torch.eye(3, device = device, dtype = dtype)                   # [3,3]
    edge_vec_R = (edge_vec[:, [1,2,0]] @ o3.Irreps("1x1o").D_from_matrix(R))[:,[2,0,1]]
    # edge_vec_R = edge_vec @ R[[1,2,0]][:, [1,2,0]]
    # edge_vec_R = edge_vec @ R
    # edge_vec_R = -edge_vec                        # 向量右乘 R

    D_in  = o3.Irreps(irreps_in).D_from_matrix(R)         # [dim_in,  dim_in]
    D_out = o3.Irreps(irreps_out).D_from_matrix(R)        # [dim_out, dim_out]

    fea_edge_R = fea_edge @ D_in

    # 旋转后的输出
    D_dict_R = compute_D(edge_vec_R, l_max=7)
    out_R, gate_R = model(fea_edge_R, edge_vec_R, D_dict_R, fea_weight)

    # -------- 检查等变性：out @ D_out(R) == out_R --------
    out_rot = out @ D_out
    diff = out_rot - out_R

    max_err  = diff.abs().max().item()
    mean_err = diff.abs().mean().item()

    print(f"[Equivariance] || out @ D_out(R) - out_R ||_max  = {max_err:.3e}")
    print(f"[Equivariance] || out @ D_out(R) - out_R ||_mean = {mean_err:.3e}")

    # 阈值可按需要调紧/放宽（GPU/CPU、随机权重均会影响数值）
    atol, rtol = 5e-5, 1e-4
    ok = torch.allclose(out_rot, out_R, atol=atol, rtol=rtol)
    assert ok, f"等变性未通过：max={max_err:.3e}, mean={mean_err:.3e}, atol={atol}, rtol={rtol}"
    print("✓ SO2_Convolution 严格旋转等变性测试通过")

if __name__ == "__main__":
    check_equivariance_SO2Conv()
