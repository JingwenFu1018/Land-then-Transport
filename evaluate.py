#!/usr/bin/env python
"""
evaluate.py: LTT test-set decoding and evaluation

功能：
1)  加载指定或自动寻找到的 checkpoint。
2)  加载数据集的 "test" split。
3)  遍历 --test_snr_db_list 中定义的每个 SNR (dB)。
4)  在完整测试集上运行解码，统计 PSNR / MS-SSIM(dB) / (可选)LPIPS。
5)  打印每个 SNR 的平均指标，并可选记录到 Weights & Biases。

Rayleigh 信道采用“复观测 → 预均衡 → 等效 AWGN”的评测路径；
模型可以是实通道 (C) 或复通道 (2C)，脚本会自动适配。

用法示例：

# 在 DIV2K (Rayleigh) 上测试，显式指定 ckpt，评估 5, 10, 15 dB（允许模型为实通道或复通道）
python evaluate.py \
  --dataset div2k \
  --output_dir output \
  --checkpoint outputs/div2k/ckpt_best_...pth \
  --test_snr_db_list "5,10,15" \
  --channel rayleigh \
  --csi_mode perfect \
  --use_complex_channels False \
  --wandb_project cfm-test \
  --wandb_name div2k-rayleigh-test
"""

import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as std_tqdm

try:
    import wandb
except ImportError:  # Optional experiment tracking
    wandb = None

from flow_matching.awgn_decode import (
    decode_from_channel_sigma,  # 允许直接按 σ_ch 解码（支持逐样本）
    snr_db_to_sigma_ch,
)

# ----------------------------------------------------------------------------
# 依赖模块（与你的项目结构一致）
# ----------------------------------------------------------------------------
from flow_matching.datasets.image_datasets import (
    get_div2k_test_transform,
    get_image_dataset,
    get_test_transform,
)
from flow_matching.metrics import LPIPSHelper, ms_ssim_db, psnr
from flow_matching.mimo_decode import mimo2x2_to_awgn
from flow_matching.models import UNetModel
from flow_matching.rayleigh_decode import (
    rayleigh_to_awgn,  # 预均衡: (y_cplx, h_hat, sigma_ch) -> (y_awgn, sigma_eff)
)
from flow_matching.utils import model_size_summary, set_seed
from ltt.checkpoint import load_model_from_checkpoint, resolve_checkpoint
from ltt.config import EvaluateConfig, parse_dataclass, parse_snr_list

tqdm = partial(std_tqdm, dynamic_ncols=True)

# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def _get_tf_and_root(args: EvaluateConfig, train: bool):
    """Select test transforms and the caller-configured portable data root."""
    is_div2k = args.dataset.lower() in ("div2k", "div2k_hr", "div2k-hr")
    transform = get_div2k_test_transform() if is_div2k else get_test_transform()
    return transform, Path(args.data_root).expanduser()


def _estimate_power(dataloader: DataLoader, n_batches=50, device="cuda"):
    """估计数据集的平均功率 E[|x|^2]（目前已不再使用，保留做参考）。"""
    s, c = 0.0, 0
    it = iter(dataloader)
    for i in range(min(len(dataloader), n_batches)):
        try:
            (xb, _) = next(it)
        except StopIteration:
            break
        xb = xb.to(device).float()
        s += xb.pow(2).mean().item()
        c += 1
    return s / max(c, 1)


# ----------------------------------------------------------------------------
# 评测：从观测回积分到干净图像并统计指标（逐样本功率版本）
# ----------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_decode_metrics(
    flow: UNetModel,
    loader: DataLoader,
    snr_db: float,
    sigma_max: float,
    schedule: str,
    device: torch.device,
    steps: int = 201,
    method: str = "midpoint",
    max_batches: int | None = None,
    use_lpips: bool = True,
    *,
    channel: str = "awgn",
    use_complex_channels: bool = False,
    csi_mode: str = "perfect",
    csi_noise_std: float = 0.0,
    mimo_corr_rho: float = 0.0,
):
    """
    在给定 data loader 上做“信道观测->回积分解码”，计算 PSNR / MS-SSIM(dB) / (可选)LPIPS。

    关键：
    - 逐样本功率 Px_i -> 逐样本或逐对 sigma_ch；
    - AWGN / Rayleigh / MIMO 统一使用 decode_from_channel_sigma；
    - Rayleigh / MIMO 通过预均衡转为等效 AWGN。
    """
    flow.eval()

    lpips_helper = LPIPSHelper(device=str(device), net_type="vgg") if use_lpips else None

    def cplx_mul(x, h2):
        """x: (B,2C,H,W) or (B,2C,...) ; h2: (B,2) with (re,im)."""
        B, C2 = x.shape[:2]
        C = C2 // 2
        xr, xi = x[:, :C], x[:, C:]
        hr = h2[:, 0].view(B, 1, *([1] * (x.dim() - 2)))
        hi = h2[:, 1].view(B, 1, *([1] * (x.dim() - 2)))
        yr = hr * xr - hi * xi
        yi = hr * xi + hi * xr
        return torch.cat([yr, yi], dim=1)

    sum_psnr = sum_msssim = sum_lpips = sum_noisy_psnr = 0.0
    n_imgs = 0
    it = iter(loader)
    steps_run = 0
    limit = len(loader) if max_batches is None else min(len(loader), max_batches)

    # ===== 新增：用于控制“每个 SNR 只打印一次 ODE 参数” =====
    printed_ode = False
    # ======================================================

    # 模型输入通道数（若可得）
    model_in_channels = getattr(flow, "in_channels", None)

    pbar = tqdm(total=limit, desc=f"Eval {snr_db}dB")
    while steps_run < limit:
        try:
            x1, y_lbl = next(it)
        except StopIteration:
            break
        steps_run += 1
        pbar.update(1)

        x1 = x1.to(device)
        y_lbl = y_lbl.to(device)
        B, C, *spatial = x1.shape

        # -----------------------------
        # 1) 逐样本功率 Px_i（每张图像一个值）-> sigma_ch_i
        # -----------------------------
        Px = x1.view(B, -1).pow(2).mean(dim=1)                  # (B,)
        sigma_ch = snr_db_to_sigma_ch(snr_db, signal_power=Px)  # (B,)

        # broadcast 用的噪声 std（AWGN / Rayleigh 用到）
        sigma_noise = sigma_ch.view(B, *([1] * (x1.dim() - 1)))  # (B,1,1,1)

        # -----------------------------
        # 2) 构造信道观测 & 预均衡
        # -----------------------------
        if channel == "rayleigh":
            # 1) 构造复格式观测（仅用于仿真/均衡）
            x1c = torch.cat([x1, torch.zeros_like(x1)], dim=1)       # (B,2C,...)
            h = torch.randn(B, 2, device=device) / (2.0 ** 0.5)     # h ~ CN(0,1)

            noise_cplx = torch.randn_like(x1c)
            y_cplx = cplx_mul(x1c, h) + sigma_noise * noise_cplx    # (B,2C,...)

            # 2) CSI: perfect / noisy
            if csi_mode == "noisy":
                h_hat = h + csi_noise_std * torch.randn_like(h)
            else:
                h_hat = h

            # 3) Rayleigh -> 等效 AWGN
            y_awgn, sigma_eff = rayleigh_to_awgn(y_cplx, h_hat, sigma_ch)  # y_awgn: (B,C,...), sigma_eff: (B,)

            # 4) 适配模型通道数
            if (model_in_channels is not None) and (model_in_channels == 2 * C):
                y_in = torch.cat([y_awgn, torch.zeros_like(y_awgn)], dim=1)  # (B,2C,...)
            else:
                y_in = y_awgn

            sigma_for_decode = sigma_eff          # (B,)
            y_obs_eval = y_awgn                   # Noisy PSNR 用等效 AWGN 观测

        elif channel == "mimo":
            # 2×2 MIMO：每 2 张图像一对
            if B < 2:
                # 这一批不足 2 张，跳过
                continue
            if B % 2 != 0:
                # 丢掉最后一个，使得 B 为偶数（x1 / y_lbl / sigma_ch 同时截断）
                x1 = x1[:-1]
                y_lbl = y_lbl[:-1]
                sigma_ch = sigma_ch[:-1]
                B, C, *spatial = x1.shape
                if B < 2:
                    continue

            # 每对的信道噪声 std（可以简单取两张的平均）
            sigma_pair = 0.5 * (sigma_ch[0::2] + sigma_ch[1::2])  # (B_pairs,)

            # 通过 2×2 MIMO + MMSE 预均衡，得到等效 AWGN 观测与逐样本噪声 std
            y_awgn, sigma_eff = mimo2x2_to_awgn(
                x_real=x1,
                sigma_ch=sigma_pair,
                csi_mode=csi_mode,
                csi_noise_std=csi_noise_std,
                corr_rho=mimo_corr_rho,
            )  # y_awgn: (B,C,H,W), sigma_eff: (B,)

            # 适配模型通道数（与 AWGN 分支一致）
            if (model_in_channels is not None) and (model_in_channels == 2 * C):
                y_in = torch.cat([y_awgn, torch.zeros_like(y_awgn)], dim=1)
            else:
                y_in = y_awgn

            sigma_for_decode = sigma_eff      # (B,)
            y_obs_eval = y_awgn

        else:
            # AWGN：y = x + n；噪声逐样本 sigma_ch
            noise = torch.randn_like(x1)
            y_obs = x1 + sigma_noise * noise  # (B,C,...)

            # 若模型是 2C，则扩展 [实, 0]
            if (model_in_channels is not None) and (model_in_channels == 2 * C):
                y_in = torch.cat([y_obs, torch.zeros_like(y_obs)], dim=1)
            else:
                y_in = y_obs

            sigma_for_decode = sigma_ch       # (B,)
            y_obs_eval = y_obs

        # -----------------------------
        # 3) 解码：统一用“已知 sigma_ch”的接口
        # -----------------------------

        debug_this_batch = (not printed_ode)

        x_hat = decode_from_channel_sigma(
            y_obs=y_in,
            sigma_ch=sigma_for_decode,        # float 或 (B,)
            sigma_max=float(sigma_max),
            model=flow,
            schedule=str(schedule),
            labels=y_lbl,
            steps=steps,
            method=method,
            return_intermediates=False,
            debug_ode=debug_this_batch,       # 新增：传入
        )
        if not torch.is_tensor(x_hat):
            x_hat = x_hat[-1]
        x_hat = x_hat.detach()

        if debug_this_batch:
            printed_ode = True  # 以后这一 SNR 不再打印

        # 评测只看实通道
        x_hat_eval = x_hat[:, :C] if x_hat.size(1) > C else x_hat

        # -----------------------------
        # 4) 累积指标
        # -----------------------------
        n = x1.size(0)
        n_imgs += n
        noisy_psnr = psnr(y_obs_eval, x1)
        cur_psnr = psnr(x_hat_eval, x1)
        cur_msssim = ms_ssim_db(x_hat_eval, x1)
        sum_noisy_psnr += noisy_psnr * n
        sum_psnr += cur_psnr * n
        sum_msssim += cur_msssim * n

        if lpips_helper is not None:
            cur_lpips = lpips_helper(x_hat_eval, x1)
            sum_lpips += cur_lpips * n

    pbar.close()

    out = dict(
        psnr_db = sum_psnr / max(1, n_imgs),
        ms_ssim_db = sum_msssim / max(1, n_imgs),
        noisy_psnr_db = sum_noisy_psnr / max(1, n_imgs),
        n_images = n_imgs,
    )
    if use_lpips:
        out["lpips"] = sum_lpips / max(1, n_imgs)
    return out



# ----------------------------------------------------------------------------
# 主测试函数
# ----------------------------------------------------------------------------
def run_test(args: EvaluateConfig):
    print("=== Running Full Test Set Evaluation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir) / args.dataset

    # 1) 加载测试集
    test_tf, test_root = _get_tf_and_root(args, train=False)
    test_set = get_image_dataset(args.dataset, train=False, transform=test_tf, root=test_root, download=args.download)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    print(f"Loaded test set '{args.dataset}' with {len(test_set)} images.")

    # 2) 获取数据形状信息（用于模型加载的后备信息）
    try:
        sample_x, _ = test_set[0]
        C_data = sample_x.size(0)
        input_shape_data = list(sample_x.size()) # [C,H,W]
        num_classes = len(test_set.classes)
        print(f"Test data: C={input_shape_data[0]}, H={input_shape_data[1]}, W={input_shape_data[2]}, N_classes={num_classes}")
    except Exception as e:
        print(f"[Warning] Could not get dataset info: {e}. Relying fully on checkpoint.")
        C_data = None
        input_shape_data = None
        num_classes = None

    # 3) 加载模型
    ckpt_path = resolve_checkpoint(out_dir, args.checkpoint)
    print(f"[test] Loading checkpoint: {ckpt_path.name}")

    flow, ckpt_sigma_max, ckpt_schedule = load_model_from_checkpoint(
        ckpt_path, device,
        input_shape=input_shape_data,
        num_classes=num_classes
    )
    flow.eval()
    model_size_summary(flow)

    # 4) 确定 sigma_max 和 schedule（命令行覆盖 ckpt/文件名推断）
    sigma_max_use = args.sigma_max if args.sigma_max is not None else ckpt_sigma_max
    schedule_use = args.sigma_schedule or ckpt_schedule

    if sigma_max_use is None:
        raise ValueError("Cannot determine sigma_max. 请通过 --sigma_max 指定，或使用带 'smax' 片段的 ckpt 文件名。")
    print(f"[test] Using sigma_max = {sigma_max_use:.4f}, schedule = {schedule_use}")

    # 5) 仅提示通道兼容性（不再强制）
    model_in_channels = getattr(flow, "in_channels", None)
    if model_in_channels is not None and C_data is not None:
        print(f"[test] Model in_channels: {model_in_channels} | Data C: {C_data}")
        if args.channel == "rayleigh":
            if model_in_channels == 2 * C_data:
                print("[Info] Rayleigh via pre-eq: 模型为 2C，将把等效 AWGN 观测扩展为 [实, 0] 以匹配。")
            elif model_in_channels != C_data:
                print(f"[Warn] 模型输入通道数与数据不匹配 (model={model_in_channels}, data={C_data})，请确认。")
        else:
            if args.use_complex_channels and model_in_channels != 2 * C_data:
                print(f"[Warn] 你请求了 --use_complex_channels，但模型输入通道并非 2C (model={model_in_channels}).")
            if (not args.use_complex_channels) and model_in_channels != C_data:
                print(f"[Warn] 模型输入通道({model_in_channels})与数据通道({C_data})不一致。")

    # 6) 解析 SNR 列表
    snr_list = parse_snr_list(args.test_snr_db_list)
    if not snr_list:
        print("[Error] No SNRs to test. 请提供 --test_snr_db_list (如 '0,5,10')")
        return

    print(f"[test] Evaluating SNRs: {snr_list} (dB)")
    if args.channel == "rayleigh":
        ch_name = "RAYLEIGH"
    elif args.channel == "mimo":
        ch_name = "MIMO"
    else:
        ch_name = "AWGN"

    if args.channel == "mimo" and abs(float(args.mimo_corr_rho)) > 1e-12:
        print(f"[test] Using correlated 2x2 MIMO with symmetric Kronecker rho={args.mimo_corr_rho:.3f}")


    # 7) 循环评估
    all_results = []
    for snr_db in snr_list:
        print(f"\n--- Evaluating {ch_name} @ {snr_db:.1f} dB ---")

        # === 计时开始 ===
        start_time = time.time()

        metrics = _evaluate_decode_metrics(
            flow=flow,
            loader=test_loader,
            snr_db=snr_db,
            sigma_max=float(sigma_max_use),
            schedule=str(schedule_use),
            device=device,
            steps=args.ode_steps,
            method="midpoint",
            max_batches=None,  # 完整测试集
            use_lpips=args.use_lpips,
            channel=args.channel,
            use_complex_channels=args.use_complex_channels,
            csi_mode=args.csi_mode,
            csi_noise_std=args.csi_noise_std,
            mimo_corr_rho=args.mimo_corr_rho,
        )

        # === 计时结束 ===
        elapsed = time.time() - start_time

        # 打印结果
        result_str = (
            f"[TEST-DECODE (full) @ {snr_db:.1f} dB] "
            f"PSNR={metrics['psnr_db']:.3f} dB | "
            f"MS-SSIM(dB)={metrics['ms_ssim_db']:.3f} | "
            f"Noisy PSNR={metrics['noisy_psnr_db']:.3f} dB | "
            f"N={metrics['n_images']}"
            + (f" | LPIPS={metrics['lpips']:.4f}" if 'lpips' in metrics else "")
        )
        print(result_str)
        print(f"[TIME] Eval {ch_name} @ {snr_db:.1f} dB took {elapsed:.2f} seconds.")
        all_results.append(result_str)

        # 记录到 Wandb (每个 SNR 点 log 一次)
        if wandb is not None and wandb.run is not None and not wandb.run.disabled:
            log_data = {
                "test_decode/snr_db": snr_db,
                "test_decode/psnr_db": metrics["psnr_db"],
                "test_decode/ms_ssim_db": metrics["ms_ssim_db"],
                "test_decode/noisy_psnr_db": metrics["noisy_psnr_db"],
                "test_decode/n_images": metrics["n_images"],
                "test_decode/time_sec": elapsed,  # 可选：把时间也 log 到 wandb
            }
            if 'lpips' in metrics:
                log_data["test_decode/lpips"] = metrics["lpips"]
            wandb.log(log_data)

    print("\n=== Test Summary ===")
    for res_str in all_results:
        print(res_str)


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluation_args = parse_dataclass(EvaluateConfig)

    if evaluation_args.use_wandb:
        if wandb is None:
            raise RuntimeError("W&B tracking requires: pip install -e '.[tracking]'")
        wandb.init(
            project=evaluation_args.wandb_project,
            entity=evaluation_args.wandb_entity,
            name=evaluation_args.wandb_name,
            config=asdict(evaluation_args),
            save_code=False,
        )

    run_test(evaluation_args)

    if wandb is not None and wandb.run is not None:
        wandb.finish()
