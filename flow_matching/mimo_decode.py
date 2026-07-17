# file: flow_matching/mimo_decode.py
from __future__ import annotations

import math

import torch
from torch import Tensor


def _kronecker_corr_sqrt(
    rho: float,
    device: torch.device,
) -> Tensor:
    """返回 2x2 相关矩阵 [[1,rho],[rho,1]] 的 Hermitian square root。"""
    rho = float(rho)
    if abs(rho) < 1e-12:
        return torch.eye(2, device=device, dtype=torch.complex64)

    if abs(rho) >= 1.0:
        raise ValueError(f"corr_rho must satisfy |rho| < 1, got {rho}.")

    corr = torch.tensor(
        [[1.0, rho], [rho, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    eigvals, eigvecs = torch.linalg.eigh(corr)
    eigvals = torch.clamp(eigvals, min=0.0).sqrt()
    return (eigvecs @ torch.diag(eigvals) @ eigvecs.transpose(-2, -1)).to(torch.complex64)


@torch.no_grad()
def mimo2x2_to_awgn(
    x_real: Tensor,                 # (B, C, H, W) 发送的“干净图像”，实数
    sigma_ch: float | Tensor,       # 信道 AWGN 标准差（与 AWGN / Rayleigh 分支同源）
    csi_mode: str = "perfect",      # {"perfect", "noisy"}
    csi_noise_std: float = 0.0,     # CSI 噪声 std（仅当 csi_mode="noisy" 时生效）
    corr_rho: float = 0.0,          # 对称 Kronecker 相关系数（Tx/Rx 取相同 rho）
) -> tuple[Tensor, Tensor]:
    """
    2×2 MIMO -> AWGN-like 观测：MMSE 预均衡版本。

    模型：
        每 2 张图像组成一对，一起通过 2×2 MIMO 信道：
            y = H_true x + n,   H_true ~ CN(0, I), n ~ CN(0, sigma_ch^2 I)

        若 corr_rho > 0，则采用对称 Kronecker 相关模型：
            H_true = R_rx^{1/2} H_w R_tx^{1/2},
            R_tx = R_rx = [[1, rho], [rho, 1]]

        然后用 MMSE 均衡器：
            G_mmse = (H_hat^H H_hat + sigma^2 I)^(-1) H_hat^H
            z      = G_mmse y

        对每个 Tx 流（两路）：
            z[i] ≈ x[i] + n_eff(i)
            sigma_eff(i) = sigma_ch * ||row_i(G_mmse)||_2

    输入:
        x_real  : (B, C, H, W)，B 必须为偶数（每 2 个样本组成一对）
        sigma_ch: 标量或 (B_pairs,)；通常来自 snr_db_to_sigma_ch(...)
        csi_mode: "perfect" 或 "noisy"
        csi_noise_std: 若 noisy，则 H_hat = H_true + CN(0, csi_noise_std^2)
        corr_rho: 对称 Kronecker 相关系数；rho=0 退化为 i.i.d. Rayleigh MIMO

    输出:
        y_awgn     : (B, C, H, W)，等效 AWGN-like 观测（实数）
        sigma_eff  : (B,)         ，每个样本对应的等效噪声标准差
    """
    device, dtype = x_real.device, x_real.dtype
    B, C, H, W = x_real.shape

    if B % 2 != 0:
        raise ValueError(f"mimo2x2_to_awgn: batch 大小必须为偶数，当前 B={B}")

    B_pairs = B // 2                     # 每 2 张图像一对
    # 重排成 (B_pairs, N_tx=2, C, H, W)
    x_pair = x_real.view(B_pairs, 2, C, H, W)
    # 转成复数类型，虚部初始为 0
    x_pair_c = x_pair.to(torch.complex64)

    # --- sigma_ch -> (B_pairs,) ---
    if not torch.is_tensor(sigma_ch):
        sigma_vec = torch.full((B_pairs,), float(sigma_ch),
                               device=device, dtype=torch.float32)
    else:
        sigma_vec = sigma_ch.to(device=device, dtype=torch.float32).view(-1)
        if sigma_vec.numel() == 1:
            sigma_vec = sigma_vec.repeat(B_pairs)
        elif sigma_vec.numel() != B_pairs:
            raise ValueError(
                f"mimo2x2_to_awgn: sigma_ch.numel()={sigma_vec.numel()} "
                f"与 pair 数量 B_pairs={B_pairs} 不一致。"
            )

    # --- 采样真实 2×2 MIMO 信道 H_true ---
    H_re = torch.randn(B_pairs, 2, 2, device=device) / math.sqrt(2.0)
    H_im = torch.randn(B_pairs, 2, 2, device=device) / math.sqrt(2.0)
    H_iid = H_re + 1j * H_im                       # (B_pairs, N_rx=2, N_tx=2)

    if abs(float(corr_rho)) > 1e-12:
        R_sqrt = _kronecker_corr_sqrt(float(corr_rho), device=device)
        H_true = R_sqrt.unsqueeze(0) @ H_iid @ R_sqrt.unsqueeze(0)
    else:
        H_true = H_iid

    # --- 通过 MIMO 信道：y = H_true x + n ---
    # x_pair_c: (B_pairs, N_tx=2, C, H, W)
    # y:       (B_pairs, N_rx=2, C, H, W)
    y = torch.einsum("bnm,bmchw->bnchw", H_true, x_pair_c)

    # 复高斯噪声 n ~ CN(0, sigma_ch^2 I)
    noise_re = torch.randn_like(y.real)
    noise_im = torch.randn_like(y.real)
    sigma_map = sigma_vec.view(B_pairs, 1, 1, 1, 1)
    noise = (noise_re + 1j * noise_im) * (sigma_map / math.sqrt(2.0))
    y = y + noise

    # --- CSI 估计 H_hat ---
    if csi_mode == "noisy" and csi_noise_std > 0.0:
        nH_re = torch.randn_like(H_re) * (csi_noise_std / math.sqrt(2.0))
        nH_im = torch.randn_like(H_im) * (csi_noise_std / math.sqrt(2.0))
        H_hat = H_true + (nH_re + 1j * nH_im)
    else:
        H_hat = H_true

    # --- MMSE 均衡：G = (H^H H + sigma^2 I)^(-1) H^H ---
    H_herm = H_hat.conj().transpose(-2, -1)        # (B_pairs, N_tx, N_rx)
    HH = H_herm @ H_hat                            # (B_pairs, N_tx, N_tx)
    identity = torch.eye(2, device=device, dtype=torch.complex64).expand(B_pairs, -1, -1)
    sigma2 = sigma_vec ** 2                        # (B_pairs,)
    A = HH + sigma2.view(B_pairs, 1, 1) * identity  # (B_pairs, 2, 2)
    G = torch.linalg.solve(A, H_herm)              # (B_pairs, N_tx, N_rx)

    # --- 预均衡输出 z = G y ---
    # z: (B_pairs, N_tx=2, C, H, W)
    z = torch.einsum("bnm,bmchw->bnchw", G, y)

    # 等效噪声 std: sigma_eff = sigma_ch * ||G 的每一行||_2
    row_norm_sq = (G.abs() ** 2).sum(dim=-1)       # (B_pairs, N_tx)
    sigma_eff = sigma_vec.view(B_pairs, 1) * torch.sqrt(row_norm_sq)  # (B_pairs, 2)

    # 只取实部作为 AWGN-like 观测
    z_real = z.real.to(dtype)                      # (B_pairs, 2, C, H, W)

    # 展开回 (B, C, H, W)，与原始 x_real 样本一一对应
    B_eff = B_pairs * 2
    y_awgn = z_real.view(B_eff, C, H, W)           # (B, C, H, W)
    sigma_eff_flat = sigma_eff.reshape(B_eff)      # (B,)

    return y_awgn, sigma_eff_flat
