from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from torch.amp import GradScaler
except ImportError:  # PyTorch 2.1 compatibility
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm as std_tqdm

try:
    import wandb
except ImportError:  # Optional experiment tracking
    wandb = None

from flow_matching.awgn_decode import decode_from_channel, decode_from_channel_sigma, snr_db_to_sigma_ch
from flow_matching.datasets.image_datasets import (
    get_div2k_test_transform,  # 新增
    get_div2k_train_transform,  # 新增
    get_image_dataset,
    get_test_transform,
    get_train_transform,
)
from flow_matching.metrics import LPIPSHelper, ms_ssim_db, psnr
from flow_matching.models import UNetModel
from flow_matching.rayleigh_decode import rayleigh_to_awgn  # 新
from flow_matching.schedules import dsigma_dt, sigma
from flow_matching.utils import model_size_summary, set_seed
from ltt.checkpoint import checkpoint_path, save_checkpoint
from ltt.config import TrainConfig, parse_dataclass, parse_snr_list

tqdm = partial(std_tqdm, dynamic_ncols=True)

# ---------------------------
# 工具函数：teacher 批处理 & 评估
# ---------------------------
def _teacher_batch(x_1: Tensor, sigma_max: float, schedule: str) -> tuple[Tensor, Tensor, Tensor]:
    """
    给定干净样本 x_1，生成 CFM 的 teacher 样本：
      t ~ U[ε,1], x_t = x_1 + σ(t) ε, u_t = (σ'(t)/σ(t)) (x_t - x_1)
    返回: (t, x_t, u_t)
    """
    B = x_1.size(0)
    device = x_1.device
    t = torch.rand(B, device=device, dtype=x_1.dtype).clamp_(min=1e-4, max=1.0)
    eps = torch.randn_like(x_1)
    sigma_t = sigma(t, sigma_max, kind=schedule).view(B, 1, 1, 1)
    x_t = x_1 + sigma_t * eps
    coef = (dsigma_dt(t, sigma_max, kind=schedule) /
            torch.clamp(sigma(t, sigma_max, kind=schedule), min=1e-8)).view(B, 1, 1, 1)
    u_t = coef * (x_t - x_1)
    return t, x_t, u_t


def _evaluate_loss(
    flow: UNetModel,
    loader: DataLoader,
    sigma_max: float,
    schedule: str,
    device: torch.device,
    max_batches: int | None = None,
    channel: str = "awgn",
    use_complex_channels: bool = False
) -> float:
    """验证集 loss：统一按 AWGN teacher 计算。"""
    flow.eval()
    total, n = 0.0, 0
    it = iter(loader)
    steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    for _ in range(steps):
        try:
            x_1, y = next(it)
        except StopIteration:
            break
        x_1, y = x_1.to(device), y.to(device)

        # 统一 AWGN teacher
        t, x_t, u_t = _teacher_batch(x_1, sigma_max, schedule)
        vf_t = flow(t=t, x=x_t, y=y)

        loss = F.mse_loss(vf_t, u_t, reduction="mean")
        bs = x_1.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)



# @torch.no_grad()
# def _evaluate_loss(
#     flow: UNetModel,
#     loader: DataLoader,
#     sigma_max: float,
#     schedule: str,
#     device: torch.device,
#     max_batches: Optional[int] = None,
#     channel: str = "awgn",
#     use_complex_channels: bool = False
# ) -> float:
#     flow.eval()
#     total, n = 0.0, 0
#     it = iter(loader)
#     steps = len(loader) if max_batches is None else min(len(loader), max_batches)
#     for _ in range(steps):
#         try:
#             x_1, y = next(it)
#         except StopIteration:
#             break
#         x_1, y = x_1.to(device), y.to(device)
#         if channel == "rayleigh":
#             t, x_t, u_t, h_vec, sigma_ch = teacher_batch_rayleigh(
#                 x1_real=x_1, sigma_max=sigma_max, schedule=schedule, sigma_ch=None
#             )
#             x_in = x_t if use_complex_channels else x_1         # 若你坚持实通道训练（不建议）
#             vf_t = flow(t=t, x=x_in, y=y, h=h_vec, sigma_ch=sigma_ch)
#         else:
#             t, x_t, u_t = _teacher_batch(x_1, sigma_max, schedule)
#             vf_t = flow(t=t, x=x_t, y=y)
#         loss = F.mse_loss(vf_t, u_t, reduction="mean")
#         bs = x_1.size(0)
#         total += loss.item() * bs
#         n += bs
#     return total / max(1, n)


def _estimate_power(dataloader: DataLoader, n_batches=50, device="cuda"):
    s, c = 0.0, 0
    for i, (xb, _) in enumerate(dataloader):
        xb = xb.to(device).float()
        s += xb.pow(2).mean().item()
        c += 1
        if i + 1 >= n_batches:
            break
    return s / max(c, 1)


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
):
    """
    在给定 data loader 上做“信道观测->回积分解码”，计算 PSNR / MS-SSIM(dB) / LPIPS。
    Rayleigh 评测通过预均衡转 AWGN 后解码
    """
    flow.eval()

    # 估计数据功率，用于设置信道噪声标准差
    Px = _estimate_power(loader, n_batches=50, device=device)
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
    while steps_run < limit:
        try:
            x1, y_lbl = next(it)
        except StopIteration:
            break
        steps_run += 1

        x1 = x1.to(device)
        y_lbl = y_lbl.to(device)

        sigma_ch = snr_db_to_sigma_ch(snr_db, signal_power=Px)

        # if channel == "rayleigh":
        #     # 建议：Rayleigh 必须配合复通道
        #     assert use_complex_channels, "Rayleigh 评测需要 use_complex_channels=True"

        #     # x1 -> 复格式 [实, 虚=0]
        #     x1c = torch.cat([x1, torch.zeros_like(x1)], dim=1)

        #     # 采样 h ~ CN(0,1)  => Re/Im ~ N(0,1/2)
        #     B = x1.size(0)
        #     h = torch.randn(B, 2, device=device) / (2.0 ** 0.5)

        #     # 构造观测 y = h x1 + n （复高斯噪声，各实通道 std=σ_ch）
        #     y_obs = cplx_mul(x1c, h) + sigma_ch * torch.randn_like(x1c)

        #     # CSI：perfect 或 noisy
        #     if csi_mode == "noisy":
        #         h_hat = h + csi_noise_std * torch.randn_like(h)
        #     else:
        #         h_hat = h

        #     # 用 Rayleigh 解码器（你已在 rayleigh_decode.py 实现）
        #     x_hat = decode_from_channel_rayleigh(
        #         y=y_obs,
        #         h_hat=h_hat,
        #         sigma_ch=sigma_ch,
        #         sigma_max=float(sigma_max),
        #         model=flow,
        #         schedule=str(schedule),
        #         labels=y_lbl,
        #         steps=steps,
        #         method=method,
        #         return_intermediates=False,
        #     )
        #     if not torch.is_tensor(x_hat):
        #         x_hat = x_hat[-1]
        #     x_hat = x_hat.detach()

        #     # 仅以实部与 GT 对齐评测
        #     C = x1.size(1)
        #     x_hat_eval = x_hat[:, :C]
        #     y_obs_eval = y_obs[:, :C]

        if channel == "rayleigh":
            # 仅用于仿真的复格式观测
            x1c = torch.cat([x1, torch.zeros_like(x1)], dim=1)     # (B,2C,...)
            B = x1.size(0)
            h = torch.randn(B, 2, device=device) / (2.0 ** 0.5)    # h ~ CN(0,1)

            # y = h ⊗ x1c + n
            y_cplx = cplx_mul(x1c, h) + sigma_ch * torch.randn_like(x1c)

            # CSI: perfect or noisy
            h_hat = h + (csi_noise_std * torch.randn_like(h) if csi_mode == "noisy" else 0)

            # 预均衡 -> (y_awgn, σ_eff)
            y_awgn, sigma_eff = rayleigh_to_awgn(y_cplx, h_hat, sigma_ch)

            # 按 σ_eff 直接解码（复用 AWGN ODE）
            x_hat = decode_from_channel_sigma(
                y_obs=y_awgn, sigma_ch=sigma_eff, sigma_max=float(sigma_max),
                model=flow, schedule=str(schedule), labels=y_lbl,
                steps=steps, method=method, return_intermediates=False,
            )
            if not torch.is_tensor(x_hat):
                x_hat = x_hat[-1]
            x_hat = x_hat.detach()

            x_hat_eval = x_hat             # 实通道
            y_obs_eval = y_awgn


        else:
            # AWGN 评测（兼容：若模型是复通道，则扩成[实, 虚=0]）
            y_obs = x1 + sigma_ch * torch.randn_like(x1)
            if use_complex_channels:
                y_obs = torch.cat([y_obs, torch.zeros_like(y_obs)], dim=1)

            x_hat = decode_from_channel(
                y_obs=y_obs,
                snr_db=snr_db,
                sigma_max=float(sigma_max),
                model=flow,
                schedule=str(schedule),
                labels=y_lbl,
                signal_power=Px,
                steps=steps,
                method=method,
                return_intermediates=False,
            )
            if not torch.is_tensor(x_hat):
                x_hat = x_hat[-1]
            x_hat = x_hat.detach()

            if use_complex_channels:
                C = x1.size(1)
                x_hat_eval = x_hat[:, :C]
                y_obs_eval = y_obs[:, :C]
            else:
                x_hat_eval = x_hat
                y_obs_eval = y_obs

        # 累积指标
        n = x1.size(0)
        n_imgs += n

        noisy_psnr = psnr(y_obs_eval, x1)
        cur_psnr = psnr(x_hat_eval, x1)
        cur_msssim = ms_ssim_db(x_hat_eval, x1)
        sum_noisy_psnr += noisy_psnr * n
        sum_psnr += cur_psnr * n
        sum_msssim += cur_msssim * n

        if lpips_helper is not None:
            cur_lpips = lpips_helper(x_hat_eval, x1)  # 内部会放到 device
            sum_lpips += cur_lpips * n

    out = dict(
        psnr_db = sum_psnr / max(1, n_imgs),
        ms_ssim_db = sum_msssim / max(1, n_imgs),
        noisy_psnr_db = sum_noisy_psnr / max(1, n_imgs),
        n_images = n_imgs,
    )
    if use_lpips:
        out["lpips"] = sum_lpips / max(1, n_imgs)
    return out

def _get_tf_and_root(args: TrainConfig, train: bool):
    """Select dataset transforms and the caller-configured portable data root."""
    is_div2k = args.dataset.lower() in ("div2k", "div2k_hr", "div2k-hr")
    if is_div2k:
        transform = (
            get_div2k_train_transform(horizontal_flip=args.horizontal_flip)
            if train
            else get_div2k_test_transform()
        )
    else:
        transform = (
            get_train_transform(horizontal_flip=args.horizontal_flip)
            if train
            else get_test_transform()
        )
    return transform, Path(args.data_root).expanduser()


def train(args: TrainConfig):
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    # 原训练集，划分出验证集
    train_tf, train_root = _get_tf_and_root(args, train=True)
    full_train = get_image_dataset(args.dataset, train=True, transform=train_tf, root=train_root, download=args.download)

    num_classes = len(full_train.classes)
    # input_shape = full_train[0][0].size()
    input_shape = tuple(full_train[0][0].size())  # [C,H,W] -> tuple
    # if args.channel == "rayleigh" and args.use_complex_channels:
    #     input_shape[0] *= 2                      # 2C
    # input_shape = tuple(input_shape)
    # 新：不做翻倍
    # input_shape = tuple(list(full_train[0][0].size()))
    print(f"input_shape={input_shape}, num_classes={num_classes}")

    val_len = max(1, int(len(full_train) * args.val_fraction))
    train_len = len(full_train) - val_len
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(full_train, [train_len, val_len], generator=gen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # 单独的测试集
    test_tf, test_root = _get_tf_and_root(args, train=False)
    test_set = get_image_dataset(args.dataset, train=False, transform=test_tf, root=test_root, download=args.download)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # 自动估 σ_max（按最低 SNR 覆盖范围）
    if args.sigma_max is None:
        Px = _estimate_power(train_loader, n_batches=50, device=device)
        sigma_ch_max = (Px ** 0.5) * (10 ** (-args.min_snr_db / 20.0))
        args.sigma_max = 1.2 * sigma_ch_max
    print(f"[CFM-AWGN] sigma_max = {args.sigma_max:.4f}, schedule = {args.sigma_schedule}")

    # 模型 & 优化器
    flow = UNetModel(input_shape, num_channels=64, num_res_blocks=2,
                 num_classes=num_classes, class_cond=True).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=device.type == "cuda")
    print("GradScaler enabled:", scaler._enabled)
    model_size_summary(flow)

    # # (W&B ADD) 监视模型
    # if wandb is not None and wandb.run is not None:
    #     wandb.watch(flow, log="all", log_freq=args.wandb_log_freq * 10)

    best_val = float("inf")
    global_step = 0  # (W&B ADD)

    for epoch in range(args.n_epochs):
        flow.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{args.n_epochs}")
        for x_1, y in pbar:
            x_1, y = x_1.to(device), y.to(device)
            # if args.channel == "rayleigh":
            #     # 复通道 or 仍用实通道由 use_complex_channels 决定
            #     t, x_t, u_t, h_vec, sigma_ch = teacher_batch_rayleigh(
            #         x1_real=x_1,
            #         sigma_max=args.sigma_max,
            #         schedule=args.sigma_schedule,
            #         sigma_ch=None,                 # 也可自定义采样策略
            #     )
            #     # forward 带入 (h, sigma_ch)
            #     vf_t = flow(t=t, x=x_t if args.use_complex_channels else x_1,
            #                 y=y, h=h_vec, sigma_ch=sigma_ch)
            # else:
            #     # 原 AWGN teacher
            #     t, x_t, u_t = _teacher_batch(x_1, args.sigma_max, args.sigma_schedule)
            #     vf_t = flow(t=t, x=x_t, y=y)

            # 训练循环中，统一：
            t, x_t, u_t = _teacher_batch(x_1, args.sigma_max, args.sigma_schedule)
            vf_t = flow(t=t, x=x_t, y=y)      # 不再传 h/sigma_ch 条件

            loss = F.mse_loss(vf_t, u_t)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # (W&B ADD) 记录训练 loss
            global_step += 1
            if wandb is not None and wandb.run is not None and global_step % args.wandb_log_freq == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "epoch": epoch + (global_step / len(train_loader)),
                    "step": global_step
                })

                # 验证（按 eval_every）
        if (epoch + 1) % args.eval_every == 0:
            # 1) 计算并打印验证 loss
            val_loss = _evaluate_loss(
                flow, val_loader, sigma_max=args.sigma_max, schedule=args.sigma_schedule,
                device=device, max_batches=args.max_val_batches,
                channel=args.channel, use_complex_channels=args.use_complex_channels
            )
            print(f"[VAL] epoch {epoch+1}  loss={val_loss:.6f}")

            # 2) 记录到 W&B
            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "val/loss": val_loss,
                    "epoch": epoch + 1,
                    "step": global_step
                })

            # 3) 保存 best ckpt
            if args.save_best and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    checkpoint_path(
                        output_dir, "ckpt_best", epoch + 1,
                        args.sigma_max, args.learning_rate,
                        args.sigma_schedule, args.channel
                    ),
                    flow, input_shape, num_classes, args.sigma_max, args.sigma_schedule, epoch + 1, val_loss
                )
                print(f"[save] New best model at epoch {epoch+1} with val loss {val_loss:.6f}")

            # 4) （可选）仅在验证轮做一次快速“解码评测”（支持多个 SNR）
            val_snr_list: list[float] = []
            if args.snr_db is not None:
                val_snr_list.append(float(args.snr_db))
            if args.test_snr_db_list:
                val_snr_list.extend(parse_snr_list(args.test_snr_db_list))

            if val_snr_list:
                for snr_db in val_snr_list:
                    eval_dec = _evaluate_decode_metrics(
                        flow=flow,
                        loader=val_loader,
                        snr_db=snr_db,
                        sigma_max=args.sigma_max,
                        schedule=args.sigma_schedule,
                        device=device,
                        steps=args.ode_steps,
                        method="midpoint",
                        max_batches=min(10, args.max_val_batches),  # 验证少量 batch，加速
                        channel=args.channel,
                        use_complex_channels=args.use_complex_channels,
                        csi_mode=args.csi_mode,
                        csi_noise_std=args.csi_noise_std,
                    )
                    print(
                        f"[VAL-DECODE e{epoch+1} @ {snr_db:.1f} dB] "
                        f"PSNR={eval_dec['psnr_db']:.3f} dB | "
                        f"MS-SSIM(dB)={eval_dec['ms_ssim_db']:.3f} | "
                        f"Noisy PSNR={eval_dec['noisy_psnr_db']:.3f} dB | "
                        f"N={eval_dec['n_images']}"
                        + (f" | LPIPS={eval_dec['lpips']:.4f}" if 'lpips' in eval_dec else "")
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log({
                            f"val_decode/psnr_db@{snr_db}dB": eval_dec["psnr_db"],
                            f"val_decode/ms_ssim_db@{snr_db}dB": eval_dec["ms_ssim_db"],
                            f"val_noisy/noisy_psnr_db@{snr_db}dB": eval_dec["noisy_psnr_db"],
                            f"val_decode/lpips@{snr_db}dB": eval_dec["lpips"],
                            "epoch": epoch + 1,
                            "step": global_step,
                        })
            else:
                print("[VAL-DECODE] 跳过解码评测（未指定 args.snr_db 或 test_snr_db_list）。")

    if args.save_last:
        save_checkpoint(
            checkpoint_path(
                output_dir,
                "ckpt_last",
                args.n_epochs,
                args.sigma_max,
                args.learning_rate,
                args.sigma_schedule,
                args.channel,
            ),
            flow,
            input_shape,
            num_classes,
            args.sigma_max,
            args.sigma_schedule,
            args.n_epochs,
            None,
        )

    print("=== TEST ===")
    test_loss = _evaluate_loss(
        flow, test_loader, sigma_max=args.sigma_max, schedule=args.sigma_schedule,
        device=device, max_batches=None,
        channel=args.channel, use_complex_channels=args.use_complex_channels,
    )
    print(f"[TEST] final loss={test_loss:.6f}")
    if wandb is not None and wandb.run is not None:
        wandb.log({"test/loss": test_loss, "step": global_step})

    # === 在测试集上做【全量】解码评测，可一次性测多个 SNR ===
    snr_list: list[float] = []
    if args.snr_db is not None:
        snr_list.append(float(args.snr_db))
    if args.test_snr_db_list:
        snr_list.extend(parse_snr_list(args.test_snr_db_list))

    if snr_list:
        for snr_db in snr_list:
            test_dec = _evaluate_decode_metrics(
                flow=flow, loader=test_loader, snr_db=snr_db,
                sigma_max=args.sigma_max, schedule=args.sigma_schedule,
                device=device, steps=args.ode_steps, method="midpoint",
                max_batches=None,
                channel=args.channel,
                use_complex_channels=args.use_complex_channels,
                csi_mode=args.csi_mode,
                csi_noise_std=args.csi_noise_std,
            )
            print(
                f"[TEST-DECODE (full) @ {snr_db:.1f} dB] "
                f"PSNR={test_dec['psnr_db']:.3f} dB | "
                f"MS-SSIM(dB)={test_dec['ms_ssim_db']:.3f} | "
                f"Noisy PSNR={test_dec['noisy_psnr_db']:.3f} dB | "
                f"N={test_dec['n_images']}"
                + (f" | LPIPS={test_dec['lpips']:.4f}" if 'lpips' in test_dec else "")
            )
            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "test_decode/snr_db": snr_db,
                    "test_decode/psnr_db": test_dec["psnr_db"],
                    "test_decode/ms_ssim_db": test_dec["ms_ssim_db"],
                    "test_decode/noisy_psnr_db": test_dec["noisy_psnr_db"],
                    **({"test_decode/lpips": test_dec["lpips"]} if "lpips" in test_dec else {}),
                    "test_decode/n_images": test_dec["n_images"],
                    "step": global_step
                })
    else:
        print("[TEST-DECODE] 跳过解码评测（未指定 args.snr_db 或 test_snr_db_list）。")

    # 也写一份标准 ckpt（便于下游加载）
    save_checkpoint(
        checkpoint_path(output_dir, "ckpt", args.n_epochs, args.sigma_max, args.learning_rate, args.sigma_schedule, args.channel),
        flow, input_shape, num_classes, args.sigma_max, args.sigma_schedule
    )


# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    train_args = parse_dataclass(TrainConfig)

    if train_args.use_wandb:
        if wandb is None:
            raise RuntimeError("W&B tracking requires: pip install -e '.[tracking]'")
        wandb.init(
            project=train_args.wandb_project,
            entity=train_args.wandb_entity,
            name=train_args.wandb_name,
            config=asdict(train_args),
            save_code=False,
        )

    train(train_args)

    if wandb is not None and wandb.run is not None:
        wandb.finish()
