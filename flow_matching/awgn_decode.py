# file: flow_matching/awgn_decode.py
from __future__ import annotations

import torch
from torch import Tensor

from flow_matching.schedules import t_from_sigma_ch
from flow_matching.solver import ModelWrapper, ODESolver


@torch.no_grad()
def snr_db_to_sigma_ch(
    snr_db: float,
    signal_power: float | Tensor = 1.0,
) -> float | Tensor:
    """
    将 SNR(dB) 映射为 AWGN 的噪声标准差 σ_ch。

    假设:
        SNR_lin = P_x / σ_ch^2
        =>  σ_ch = sqrt(P_x / SNR_lin) = sqrt(P_x) * 10^(-SNR_dB/20)

    参数:
        snr_db: 标量 SNR(dB)。
        signal_power:
            - 若为 float: 视为整批数据统一功率 P_x，返回 float。
            - 若为 Tensor (B,): 逐样本功率 P_x(i)，返回 Tensor (B,) 的 σ_ch(i)。

    注意:
        decode_from_channel(...) 仍然用标量 signal_power（默认 1.0），
        所以原有调用行为保持不变。
    """
    snr_lin = 10.0 ** (snr_db / 10.0)

    # signal_power 既可以是标量也可以是 Tensor；
    # 下面表达式会自动广播，返回类型随 signal_power 而定。
    sigma = (signal_power ** 0.5) * (snr_lin ** -0.5)
    return sigma



class _WrappedModel(ModelWrapper):
    """把 flow model 封装进 solver 所需的接口。"""
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        # 透传给 UNet：注意 t 形状与模型的期望一致（常见为 [B] 或 [B,1]）
        return self.model(x=x, t=t, **extras)


@torch.no_grad()
def solve_ode_from(
    x_t0: Tensor,
    t0: float,
    model,
    t_end: float = 0.0,
    labels: Tensor | None = None,
    steps: int = 501,
    method: str = "midpoint",
    return_intermediates: bool = False,
    debug_ode: bool = False,   # 新增：是否打印 t0 / t_end / dt
    **conds,
) -> Tensor:
    assert 0.0 <= float(t0)  <= 1.0, f"t0={t0} should be in [0,1]"
    assert 0.0 <= float(t_end) <= 1.0, f"t_end={t_end} should be in [0,1]"
    device, dtype = x_t0.device, x_t0.dtype

    # === 关键兜底：端点几乎相等 -> 直接返回，避免 time_grid 出现重复点 ===
    if abs(float(t_end) - float(t0)) < 1e-12:
        return [x_t0] if return_intermediates else x_t0

    wrapped = _WrappedModel(model)
    solver  = ODESolver(wrapped)

    kwargs = dict(conds)
    if labels is not None and "y" not in kwargs:
        kwargs["y"] = labels

    # 固定步长；steps 至少 2
    steps = max(int(steps), 2)
    dt = abs(float(t0) - float(t_end)) / steps

    # ====== 新增：需要时打印本次 ODE 的时间步信息 ======
    if debug_ode:
        print(f"[ODE] t0={t0:.6f}, t_end={t_end:.6f}, dt={dt:.6e}, steps={steps}")
    # =================================================

    # print(f'solve_ode_from: t0={t0}, t_end={t_end}, dt={dt}, steps={steps}')

    # === 用严格单调的 time_grid；轨迹模式 steps+1，否则只给两端点 ===
    if return_intermediates:
        time_grid = torch.linspace(float(t0), float(t_end), steps + 1, device=device, dtype=dtype)
    else:
        time_grid = torch.tensor([float(t0), float(t_end)], device=device, dtype=dtype)

    # 若因为浮点误差导致相邻差为 0，给一个极小扰动，保证严格单调
    dif = time_grid[1:] - time_grid[:-1]
    if (dif <= 0).all() or (dif >= 0).all():
        if (dif.abs() < 1e-15).any():
            eps = torch.linspace(0, 1e-6, time_grid.numel(), device=device, dtype=dtype)
            time_grid = time_grid + (eps if time_grid[-1] > time_grid[0] else -eps)

    # print(time_grid)

    x_end = solver.sample(
        x_init=x_t0,
        step_size=dt,
        method=method,
        time_grid=time_grid,
        return_intermediates=return_intermediates,
        **kwargs,
    )
    return x_end




@torch.no_grad()
def decode_from_channel(
    y_obs: Tensor,                 # 观测：x1 + n，形状 (B,C,H,W)，与训练尺度一致
    snr_db: float,
    sigma_max: float,              # 训练/部署共用的 σ_max，要求 σ_max >= σ_ch
    model,                         # 你的 UNet / 向量场网络
    schedule: str = "sqrt",        # 与训练一致："sqrt" 或 "linear"
    labels: Tensor | None = None,
    signal_power: float = 1.0,     # 你的数据功率 P_x（若数据已单位化，保持 1.0）
    steps: int = 501,
    method: str = "midpoint",
    return_intermediates: bool = False,
) -> Tensor:
    """
    给定接收端观测 y 和 SNR(dB)，计算 t* 并从 y 作为 x(t*) 回积分到 0。
    返回 x_hat0（或全路径）。
    """
    sigma_ch = snr_db_to_sigma_ch(snr_db, signal_power=signal_power)
    if sigma_ch > sigma_max + 1e-8:
        raise ValueError(
            f"sigma_ch={sigma_ch:.4f} exceeds sigma_max={sigma_max:.4f}. "
            f"请增大训练/部署使用的 --sigma_max 或放宽最低 SNR。"
        )
    t0 = float(t_from_sigma_ch(sigma_ch, sigma_max, kind=schedule))
    t0 = max(0.0, min(1.0, t0))  # clamp 到合法区间

    # # ==========================================================
    # print(f"[Debug] SNR: {snr_db:.2f} dB  =>  sigma_ch: {sigma_ch:.4f}  =>  t0: {t0:.4f}")
    # # ==========================================================

    # 模型切 eval，保证 BN/Dropout 等一致
    training = model.training
    model.eval()
    try:
        sol = solve_ode_from(
            x_t0=y_obs, t0=t0, t_end=0.0, model=model, labels=labels,
            steps=steps, method=method, return_intermediates=return_intermediates, 
        )
    finally:
        model.train(training)
    return sol


# ========== used for rayleigh channel ==========
@torch.no_grad()
def decode_from_channel_sigma(
    y_obs: Tensor,
    sigma_ch: float | Tensor,
    sigma_max: float,
    model,
    schedule: str = "sqrt",
    labels: Tensor | None = None,
    steps: int = 501,
    method: str = "midpoint",
    return_intermediates: bool = False,
    debug_ode: bool = False,   # 新增：控制是否打印 ODE 参数
) -> Tensor:
    """
    已知 AWGN 噪声标准差 sigma_ch（可逐样本）的解码接口。

    参数:
        y_obs      : (B, C_or_2C, H, W) 信道观测。
        sigma_ch   : float 或 Tensor(B,) —— 每个样本对应的信道噪声 std。
        sigma_max  : 训练/部署共用的 σ_max，要求 σ_max >= “典型信道噪声水平”。
        model      : UNet / 向量场模型。
        schedule   : "sqrt" / "linear" 等，与训练一致。
        labels     : (B,) 类别标签（若模型 class_cond=True）。
        steps      : ODE 步数。
        method     : ODE 积分方法。
        return_intermediates:
            - False: 返回 x_hat0
            - True : 返回整个时间轨迹列表 [x_tk]

    说明:
        - 若 sigma_ch 为标量，则视为整 batch 统一噪声；
        - 若为 Tensor(B,)，则 batch 内每个样本有自己的 σ_ch(i)，
          但积分起点 t0 统一使用 mean(sigma_ch) 对应的时间；
          逐样本 sigma_ch 仍通过条件 sigma_ch=sigma_vec 传给模型。
    """
    B = y_obs.shape[0]

    # --- 1) 统一为 (B,) 的 sigma_vec ---
    if not torch.is_tensor(sigma_ch):
        sigma_vec = torch.full(
            (B,),
            float(sigma_ch),
            device=y_obs.device,
            dtype=y_obs.dtype,
        )
    else:
        sigma_vec = sigma_ch.to(device=y_obs.device, dtype=y_obs.dtype).view(-1)
        if sigma_vec.numel() == 1:
            sigma_vec = sigma_vec.repeat(B)
        elif sigma_vec.numel() != B:
            raise ValueError(
                f"decode_from_channel_sigma: sigma_ch.numel()={sigma_vec.numel()} "
                f"与 batch 大小 B={B} 不匹配。"
            )

    # --- 2) 用 batch 平均 sigma_scalar 计算统一的 t0 ---
    sigma_scalar = float(sigma_vec.mean().item())
    t0 = float(t_from_sigma_ch(min(sigma_scalar, sigma_max),
                               float(sigma_max), kind=schedule))
    t0 = max(0.0, min(1.0, t0))

    # === 关键兜底：
    # 若 t0 已经非常接近“无噪声端”（例如 σ(t=1)=0），则认为无需解码，直接返回 y_obs。
    if abs(1.0 - t0) < 1e-12:
        return y_obs if not return_intermediates else [y_obs]

    # --- 3) 调 ODE 求解器回积分 ---
    training = model.training
    model.eval()
    try:
        sol = solve_ode_from(
            x_t0=y_obs,
            t0=t0,
            t_end=0.0,
            model=model,
            labels=labels,
            steps=steps,
            method=method,
            return_intermediates=return_intermediates,
            # 把逐样本 sigma_vec 作为连续条件传给 UNet.forward(..., sigma_ch=...)
            sigma_ch=sigma_vec,
            debug_ode=debug_ode,         # 新增：把 debug 标志往下传
        )
    finally:
        model.train(training)

    return sol
