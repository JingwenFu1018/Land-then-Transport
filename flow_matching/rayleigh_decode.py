# file: flow_matching/rayleigh_decode.py
from __future__ import annotations

import torch
from torch import Tensor

from flow_matching.awgn_decode import solve_ode_from  # 复用已改好的
from flow_matching.schedules import dsigma_dt, sigma, t_from_sigma_ch

# ---------- 复数: 实-虚分离工具 ----------
# def _complex_mul_xy(x: Tensor, h_re: Tensor, h_im: Tensor) -> Tensor:
#     """
#     x: (B,2C,...)  [前 C 为实, 后 C 为虚]
#     h_re/h_im: (B,) 逐样本 block-fading 系数
#     """
#     B, C2 = x.shape[:2]
#     assert C2 % 2 == 0, "x 应为 2*C 通道（实-虚分离）"
#     C = C2 // 2
#     x_re, x_im = x[:, :C], x[:, C:]
#     shape = [B, 1] + [1]*(x.dim()-2)
#     hr = h_re.view(*shape)
#     hi = h_im.view(*shape)
#     y_re = hr * x_re - hi * x_im
#     y_im = hr * x_im + hi * x_re
#     return torch.cat([y_re, y_im], dim=1)

def _to_complex_channels(x_real: Tensor) -> Tensor:
    """把实值图像扩成复通道：x -> [x, 0]。 形状 (B, 2C, H, W)"""
    return torch.cat([x_real, torch.zeros_like(x_real)], dim=1)

# ---------- Rayleigh teacher (not used in MMSE version)----------
@torch.no_grad()
def teacher_batch_rayleigh(
    x1_real: Tensor,               # (B,C,H,W) 实值数据（想做复数就加虚部零）
    sigma_max: float,
    schedule: str,
    sigma_ch: Tensor | None = None,  # (B,)，若不提供则均匀采样 (0, sigma_max]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    返回: t, x_t (2C), u_t (2C), h_vec(B,2), sigma_ch(B,)
    对应式(1)(2)的闭式监督。
    """
    device, dtype = x1_real.device, x1_real.dtype
    B = x1_real.size(0)

    # 复通道表示
    x1 = _to_complex_channels(x1_real)  # (B,2C,H,W)

    # t ~ U[eps,1]
    t = torch.rand(B, device=device, dtype=dtype).clamp_(min=1e-4, max=1.0)
    sig_t = sigma(t, sigma_max, kind=schedule)          # (B,)
    ds_dt = dsigma_dt(t, sigma_max, kind=schedule)      # (B,)

    # h ~ CN(0,1): Re/Im ~ N(0, 1/2)
    h_re = torch.randn(B, device=device, dtype=dtype) / (2.0 ** 0.5)
    h_im = torch.randn(B, device=device, dtype=dtype) / (2.0 ** 0.5)
    h_vec = torch.stack([h_re, h_im], dim=1)           # (B,2)

    # sigma_ch
    if sigma_ch is None:
        # 简单选择：均匀采样 (0, sigma_max]，避免 0
        sigma_ch = (torch.rand(B, device=device, dtype=dtype) * (sigma_max - 1e-4) + 1e-4)
    else:
        sigma_ch = sigma_ch.to(device=device, dtype=dtype).view(B)

    # beta(t) = min{1, sigma(t)/sigma_ch}
    beta = torch.clamp(sig_t / sigma_ch, max=1.0)                         # (B,)
    beta_map = beta.view(B, 1, 1, 1)

    # mu_t = (1-beta)*x1 + beta*(h x1)
    hx1 = _complex_mul_xy(x1, h_re, h_im)                                 # (B,2C,H,W)
    mu_t = (1.0 - beta_map) * x1 + beta_map * hx1

    # x = mu_t + sigma(t) * eps
    eps = torch.randn_like(mu_t)
    sig_map = sig_t.view(B, 1, 1, 1)
    x_t = mu_t + sig_map * eps

    # beta'(t) = dsigma_dt / sigma_ch  (当 sigma(t) < sigma_ch)；否则 0
    mask = (sig_t < sigma_ch).to(dtype)                                   # (B,)
    beta_prime = (ds_dt / torch.clamp(sigma_ch, min=1e-8)) * mask         # (B,)

    # u_t = (ds/dt / s) * (x - mu_t) + beta'(t) * (h-1) * x1
    coef = (ds_dt / torch.clamp(sig_t, min=1e-8)).view(B, 1, 1, 1)
    term1 = coef * (x_t - mu_t)

    # (h-1) * x1  ==> 复乘以 (h_re-1, h_im-0)
    term2 = _complex_mul_xy(x1, h_re - 1.0, h_im) * beta_prime.view(B, 1, 1, 1)

    u_t = term1 + term2
    return t, x_t, u_t, h_vec, sigma_ch

# ---------- 解码：从 y = h x1 + n 回积分到 0 (not used in MMSE version) ----------
@torch.no_grad()
def decode_from_channel_rayleigh(
    y: Tensor | None = None,                 # (B, 2C, H, W) 复观测 [实, 虚]
    y_obs: Tensor | None = None,            # 兼容旧调用
    h_hat: Tensor | None = None,            # (B, 2)  [Re(h), Im(h)]
    sigma_ch: float | Tensor | None = None, # 标量或 (B,)
    sigma_max: float | None = None,
    model = None,
    schedule: str = "sqrt",
    labels: Tensor | None = None,           # (B,)
    steps: int = 501,
    method: str = "midpoint",
    return_intermediates: bool = False,
) -> Tensor:
    """
    Rayleigh 解码：给定复观测 y = h x1 + n, CSI 估计 h_hat, 噪声 std σ_ch，
    计算 t* = σ^{-1}(σ_ch)，以 y 作 x(t*) 回积分到 t=0。
    - y: (B,2C,H,W) 复格式 [实, 虚]；若传 y_obs 也可。
    - h_hat: (B,2) 复 CSI (Re, Im)。
    - sigma_ch: 标量或 (B,)；若标量会扩展为 batch 张量。
    """
    # -------- 参数检查与兼容 ----------
    if y is None:
        y = y_obs
    if y is None:
        raise ValueError("decode_from_channel_rayleigh: missing `y` (or `y_obs`).")
    if h_hat is None:
        raise ValueError("decode_from_channel_rayleigh: missing `h_hat`.")
    if sigma_ch is None:
        raise ValueError("decode_from_channel_rayleigh: missing `sigma_ch`.")
    if sigma_max is None:
        raise ValueError("decode_from_channel_rayleigh: missing `sigma_max`.")
    if model is None:
        raise ValueError("decode_from_channel_rayleigh: missing `model`.")

    # σ_ch 支持标量或 (B,)
    if not torch.is_tensor(sigma_ch):
        sigma_vec = torch.full((y.shape[0],), float(sigma_ch), device=y.device, dtype=y.dtype)
    else:
        sigma_vec = sigma_ch.to(device=y.device, dtype=y.dtype).view(-1)
        if sigma_vec.numel() == 1:
            sigma_vec = sigma_vec.repeat(y.shape[0])

    # -------- t* 与边界 ----------
    # 若 batch 内 σ_ch 不同，可用每个样本自己的 t*；这里采用“统一 t* = σ^{-1}(mean σ_ch)”
    sigma_scalar = float(sigma_vec.mean().item())
    if sigma_scalar > float(sigma_max) + 1e-8:
        raise ValueError(f"sigma_ch={sigma_scalar:.4f} exceeds sigma_max={float(sigma_max):.4f}.")
    t0 = float(t_from_sigma_ch(sigma_scalar, float(sigma_max), kind=schedule))
    t0 = max(0.0, min(1.0, t0))

    # -------- ODE 回积分 ----------
    training = model.training
    model.eval()
    try:
        sol = solve_ode_from(
            x_t0=y,
            t0=t0,
            t_end=0.0,                   # <--- 关键修正：指定积分终点为 1.0
            model=model,
            labels=labels,
            steps=steps,
            method=method,
            return_intermediates=return_intermediates,
            # 关键：把 CSI 与 σ_ch 作为连续条件传给 UNet.forward(h=..., sigma_ch=...)
            h=h_hat.to(device=y.device, dtype=y.dtype),
            sigma_ch=sigma_vec,
        )
    finally:
        model.train(training)

    return sol


def _complex_mul_xy(x: Tensor, h_re: Tensor, h_im: Tensor) -> Tensor:
    B, C2 = x.shape[:2]
    C = C2 // 2
    xr, xi = x[:, :C], x[:, C:]
    shape = [B, 1] + [1] * (x.dim() - 2)
    hr = h_re.view(*shape)
    hi = h_im.view(*shape)
    yr = hr * xr - hi * xi
    yi = hr * xi + hi * xr
    return torch.cat([yr, yi], dim=1)


# # ZF versison
# @torch.no_grad()
# def rayleigh_to_awgn(y_cplx: Tensor, h_hat: Tensor, sigma_ch: float | Tensor):
#     """
#     y_cplx: (B,2C,...) 复观测 [实,虚]
#     h_hat : (B,2)      [Re(h), Im(h)]
#     返回:
#       y_awgn   : (B,C,...)   预均衡后的实通道观测
#       sigma_eff: (B,)        有效噪声 std = sigma_ch / |h|
#     """
#     device, dtype = y_cplx.device, y_cplx.dtype
#     B = y_cplx.size(0)
#     h_re, h_im = h_hat[:, 0], h_hat[:, 1]
#     mag2 = (h_re*h_re + h_im*h_im).clamp_min(1e-12)   # |h|^2

#     # y_eq = (h* / |h|^2) ⊗ y
#     y_eq = _complex_mul_xy(y_cplx, h_re, -h_im)
#     y_eq = y_eq / mag2.view(B, 1, *([1]*(y_cplx.dim()-2)))

#     C = y_cplx.size(1)//2
#     y_awgn = y_eq[:, :C].contiguous()

#     if not torch.is_tensor(sigma_ch):
#         sigma_vec = torch.full((B,), float(sigma_ch), device=device, dtype=dtype)
#     else:
#         sigma_vec = sigma_ch.to(device=device, dtype=dtype).view(-1)
#         if sigma_vec.numel() == 1:
#             sigma_vec = sigma_vec.repeat(B)
#     sigma_eff = sigma_vec / torch.sqrt(mag2)          # σ_eff = σ_ch / |h|
#     return y_awgn, sigma_eff


# MMSE version 
@torch.no_grad()
def rayleigh_to_awgn(
    y_cplx: Tensor,
    h_hat: Tensor,
    sigma_ch: float | Tensor,
):
    """
    MMSE 预均衡：
        g_mmse = h* / (|h|^2 + sigma^2)
        z = g_mmse ⊗ y = alpha * x + n_eff

    参数:
        y_cplx  : (B, 2C, ...) 复观测 [实, 虚]。
        h_hat   : (B, 2)       估计的 CSI [Re(h), Im(h)]。
        sigma_ch: float 或 Tensor(B,)
                  - float      : 所有样本共用同一信道噪声 std；
                  - Tensor(B,): 逐样本噪声 std(i)。

    返回:
        y_awgn   : (B, C, ...) —— 预均衡后的“AWGN-like”实通道观测（仍带 alpha 缩放）。
        sigma_eff: (B,)        —— 等效噪声标准差
                                   = |g_mmse| * sigma_ch
                                   = (|h| / (|h|^2 + sigma^2)) * sigma_ch
    """
    device, dtype = y_cplx.device, y_cplx.dtype
    B = y_cplx.size(0)

    # h = h_re + j h_im, |h|^2
    h_re, h_im = h_hat[:, 0], h_hat[:, 1]
    mag2 = (h_re * h_re + h_im * h_im).clamp_min(1e-12)  # (B,)

    # sigma_ch -> (B,)
    if not torch.is_tensor(sigma_ch):
        sigma_vec = torch.full((B,), float(sigma_ch), device=device, dtype=dtype)
    else:
        sigma_vec = sigma_ch.to(device=device, dtype=dtype).view(-1)
        if sigma_vec.numel() == 1:
            sigma_vec = sigma_vec.repeat(B)
        elif sigma_vec.numel() != B:
            raise ValueError(
                f"rayleigh_to_awgn: sigma_ch.numel()={sigma_vec.numel()} 与 batch 大小 B={B} 不匹配。"
            )

    # 先做 h* ⊗ y
    y_conj = _complex_mul_xy(y_cplx, h_re, -h_im)  # (B, 2C, ...)

    # MMSE 分母 |h|^2 + sigma^2
    denom = (mag2 + sigma_vec ** 2).view(B, 1, *([1] * (y_cplx.dim() - 2)))

    # y_mmse = (h* / (|h|^2 + sigma^2)) ⊗ y
    y_mmse = y_conj / denom

    # 仅取实部通道作为“AWGN-like”观测（仍然带 alpha 缩放）
    C = y_cplx.size(1) // 2
    y_awgn = y_mmse[:, :C].contiguous()

    # 等效噪声 std: |g_mmse| * sigma = (|h| / (|h|^2 + sigma^2)) * sigma
    sigma_eff = (sigma_vec * torch.sqrt(mag2)) / (mag2 + sigma_vec ** 2)  # (B,)

    return y_awgn, sigma_eff





# @torch.no_grad()
# def decode_rayleigh_known_h(
#     y_cplx: Tensor, h_hat: Tensor, sigma_ch: float | Tensor,
#     sigma_max: float, model, schedule: str="sqrt",
#     labels: Tensor | None=None, steps: int=501, method: str="midpoint",
#     return_intermediates: bool=False,
# ):
#     """
#     已知 CSI 的 Rayleigh 解码：预均衡 -> (y_awgn, σ_eff) -> 直接用 AWGN ODE 解码
#     """
#     y_awgn, sigma_eff = rayleigh_to_awgn(y_cplx, h_hat, sigma_ch)

#     sigma_scalar = float(sigma_eff.mean().item())
#     # 若越过训练上限，钳到 t0=0 防止发散；推荐把 sigma_max 设得更宽裕（见第 5 节）
#     t0 = 0.0 if sigma_scalar > sigma_max + 1e-8 else float(
#         t_from_sigma_ch(sigma_scalar, float(sigma_max), kind=schedule)
#     )
#     t0 = max(0.0, min(1.0, t0))

#     training = model.training
#     model.eval()
#     try:
#         sol = solve_ode_from(
#             x_t0=y_awgn, t0=t0, t_end=0.0,
#             model=model, labels=labels, steps=steps, method=method,
#             return_intermediates=return_intermediates,
#             sigma_ch=sigma_eff,     # 把逐样本 σ_eff 作为连续条件传给 UNet
#         )
#     finally:
#         model.train(training)
#     return sol
