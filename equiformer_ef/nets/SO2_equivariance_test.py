# strict_equivariance_test.py
import torch
from e3nn import o3

# === 按需替换成你的真实导入路径 ===
from SO2_conv import SO2_EquiConv  # 需能构造模型
# =================================

def random_so3(device, dtype):
    """用 QR 分解+行列式修正生成 Haar 随机近似的 SO(3) 旋转。"""
    A = torch.randn(3, 3, device=device, dtype=dtype)
    # QR 分解（保证正交）
    Q, R = torch.linalg.qr(A)
    # 确保右手系 det=+1
    if torch.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]
    return Q  # [3,3]

@torch.no_grad()
def strict_equivariance_test():
    torch.manual_seed(42)

    # 配置
    E = 16
    fc_len_in = 64
    irreps_in  = '32x0e+16x1o+16x1e+8x2e+8x2o+8x3e+8x3o+8x4e'
    irreps_out = '128x0e+32x0o+64x1e+64x1o+32x2e+32x2o+32x3e+32x3o+32x4e'

    dim_in  = o3.Irreps(irreps_in).dim
    dim_out = o3.Irreps(irreps_out).dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dtype  = torch.float32

    # 随机输入
    fea_edge   = torch.randn(E, dim_in,  device=device, dtype=dtype)
    edge_vec   = torch.randn(E, 3,       device=device, dtype=dtype)
    fea_weight = torch.randn(E, fc_len_in, device=device, dtype=dtype)
    batch_edge = torch.zeros(E, dtype=torch.long, device=device)

    # 模型
    model = SO2_EquiConv(
        fc_len_in=fc_len_in,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        norm='e3LayerNorm',
        nonlin=True
    ).to(device)

    # 原输出
    from SO3_tools import compute_D
    D_dict = compute_D(edge_vec, l_max=7)
    out = model(fea_edge, edge_vec, D_dict, fea_weight, batch_edge)  # [E, dim_out]
    print(out[0, 128:128+32])

    # 生成随机旋转 R
    R = random_so3(device, dtype)  # [3,3]

    # 旋转输入
    edge_vec_R = (edge_vec[:, [1,2,0]] @ o3.Irreps("1x1o").D_from_matrix(R))[:,[2,0,1]]                              # [E,3]
    D_in = o3.Irreps(irreps_in).D_from_matrix(R)           # [dim_in, dim_in]
    fea_edge_R = fea_edge @ D_in                           # [E,dim_in]
    D_dict_R = compute_D(edge_vec_R, l_max=7)

    # 旋转后的前向
    out_R = model(fea_edge_R, edge_vec_R, D_dict_R, fea_weight, batch_edge)  # [E, dim_out]

    # 将原输出按 D_out(R) 旋转以对比
    D_out = o3.Irreps(irreps_out).D_from_matrix(R)         # [dim_out, dim_out]
    out_rot = out @ D_out                                  # [E, dim_out]

    # import pdb; pdb.set_trace()
    # 误差度量
    diff = (out_rot - out_R)
    max_err = diff.abs().max().item()
    mean_err = diff.abs().mean().item()

    print(f"|| out @ D_out(R) - out_R ||_max  = {max_err:.3e}")
    print(f"|| out @ D_out(R) - out_R ||_mean = {mean_err:.3e}")

    # 阈值：e3nn 浮点数误差常见 1e-5~1e-6（具体依赖实现与随机权重）
    atol = 1e-5
    rtol = 1e-5
    # 使用 allclose 断言
    ok = torch.allclose(out_rot, out_R, atol=atol, rtol=rtol)
    assert ok, f"严格等变性未通过：max_err={max_err:.3e}, mean_err={mean_err:.3e}, atol={atol}, rtol={rtol}"
    print("✓ 严格旋转等变性测试通过")

if __name__ == "__main__":
    strict_equivariance_test()
