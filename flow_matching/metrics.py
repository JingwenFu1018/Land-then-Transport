# flow_matching/metrics.py
import warnings

import torch
import torch.nn.functional as F

# （可选）抑制 torchmetrics LPIPS 的 FutureWarning（范围很窄）
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torchmetrics\.functional\.image\.lpips"
)

# ===== MS-SSIM 后端选择：优先 pytorch_msssim，其次 torchmetrics =====
_USE_PT_MSSSIM = False
_USE_TM_MSSSIM = False

try:
    from pytorch_msssim import ms_ssim as _msssim_pt
    from pytorch_msssim import ssim as _ssim_pt
    _USE_PT_MSSSIM = True
except Exception:
    try:
        # torchmetrics >= 0.11
        from torchmetrics.functional.image.ms_ssim import (
            multiscale_structural_similarity_index_measure as _msssim_tm,
        )
        from torchmetrics.functional.image.ssim import (
            structural_similarity_index_measure as _ssim_tm,
        )
        _USE_TM_MSSSIM = True
    except Exception:
        _USE_TM_MSSSIM = False

# ===== LPIPS 后端：torchmetrics（若不可用则回退到 MSE） =====
_LPIPS_AVAILABLE = True
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except Exception:
    _LPIPS_AVAILABLE = False


@torch.no_grad()
def _to_01(x: torch.Tensor) -> torch.Tensor:
    """把张量统一到 [0,1] 区间；支持输入已在 [-1,1] 或 [0,1]。"""
    if x.min() < 0:
        x = (x + 1) / 2
    return x.clamp(0, 1)


@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR（dB）。输入可为 [-1,1] 或 [0,1]。"""
    x = _to_01(x)
    y = _to_01(y)
    mse = F.mse_loss(x, y)
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


# ======== 自适应 MS-SSIM 参数计算（修正严格不等式） ========
def _auto_ms_params(h: int, w: int, win_size: int = 11, max_levels: int = 5):
    """
    根据图像短边自适应地确定窗口大小与尺度数（levels），并给出截断归一化的权重。
    约束：min(H,W) > (win_size-1) * 2^(levels-1)  —— 注意是严格大于！
    """

    min_side = int(min(h, w))

    # 规范化窗口大小：奇数、且不超过短边-1（确保严格大于时更宽松）
    win = int(win_size)
    if win % 2 == 0:
        win -= 1
    win = max(3, min(win, max(3, min_side - 1)))  # 保证 win <= min_side-1 且 >=3

    # 计算满足严格不等式的最大 L：找到最大 L 使得 (win-1)*2^(L-1) < min_side
    denom = max(1, win - 1)
    L = 1
    # 用 while 明确满足严格不等式
    while (denom * (2 ** (L - 1))) < min_side and L < 64:
        L += 1
    L -= 1  # 上一步多加了 1
    L = max(1, min(max_levels, L))

    # 经典 5 尺度权重，按需要截断并归一化
    base = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = base[:L]
    s = sum(weights)
    weights = [w_ / s for w_ in weights] if s > 0 else [1.0]

    return win, L, weights


@torch.no_grad()
def ms_ssim_db(x: torch.Tensor, y: torch.Tensor,
               *, max_levels: int = 5, win_size: int = 11) -> float:
    """
    真·MS-SSIM，返回 dB：-10*log10(1 - MS-SSIM)。
    - 自适应小尺寸：自动减少尺度数，必要时退化到单尺度 SSIM
    - 在 pytorch_msssim 与 torchmetrics 之间自动切换
    - 额外防御：若第三方库仍断言，逐级回退直到通过
    """
    x = _to_01(x)
    y = _to_01(y)
    h, w = x.shape[-2:]
    win_use, levels, weights = _auto_ms_params(h, w, win_size=win_size, max_levels=max_levels)

    if _USE_PT_MSSSIM:
        # 定义一个可回退的调用器
        def _call_pt(L):
            if L <= 1:
                return _ssim_pt(x, y, data_range=1.0, win_size=win_use, size_average=True)
            else:
                return _msssim_pt(
                    x, y, data_range=1.0, win_size=win_use,
                    weights=weights[:L], size_average=True
                )

        # 先按自适应 L 调用；若失败则逐级回退
        try:
            val = _call_pt(levels)
        except AssertionError:
            L_try = max(1, levels - 1)
            while L_try >= 1:
                try:
                    val = _call_pt(L_try)
                    break
                except AssertionError:
                    L_try -= 1
            else:
                # 极端兜底：回退到单尺度 SSIM 再不成就用 MSE 近似
                try:
                    val = _ssim_pt(x, y, data_range=1.0, win_size=win_use, size_average=True)
                except Exception:
                    mse = F.mse_loss(x, y)
                    val = (1.0 - mse).clamp(0.0, 0.999999)

    elif _USE_TM_MSSSIM:
        # torchmetrics 分支：如果只允许 1 个尺度就用 SSIM，否则传 betas 与 kernel_size
        try:
            if levels == 1:
                val = _ssim_tm(x, y, data_range=1.0, kernel_size=win_use)
            else:
                val = _msssim_tm(x, y, data_range=1.0, kernel_size=win_use, betas=tuple(weights[:levels]))
        except TypeError:
            # 兼容旧签名（不支持 kernel_size / betas），若仍因尺寸报错则降级
            try:
                val = _msssim_tm(x, y, data_range=1.0)
            except Exception:
                val = _ssim_tm(x, y, data_range=1.0, kernel_size=win_use)

    else:
        raise RuntimeError(
            "MS-SSIM requires an optional metrics backend. "
            "Install it with: pip install -e '.[metrics]'"
        )

    if isinstance(val, torch.Tensor):
        val = val.mean()
    val = torch.clamp(val, 0.0, 0.999999)
    db = -10.0 * torch.log10(1.0 - val + 1e-12)
    return float(db.item())



class LPIPSHelper:
    """
    LPIPS 评估工具（更稳健版本）：
    - 将输入强制整理成 [N,3,H,W]；
    - 将像素稳健地规范到 [-1,1]（支持 [0,1] / [0,255] / 近似 [-1,1] 等）；
    - 若 min(H,W) < 64，先双线性插值到 >=64（LPIPS 特征金字塔更稳定）。
    """
    _warned = False

    def __init__(self, device: str = "cuda", net_type: str = "vgg", min_size: int = 64):
        self.device = device
        self.min_size = int(min_size)
        self.use_lpips = _LPIPS_AVAILABLE
        if self.use_lpips:
            self.loss = LearnedPerceptualImagePatchSimilarity(
                net_type=net_type
            ).to(device).eval()
        else:
            self.loss = None

    @staticmethod
    def _to_neg1to1(z: torch.Tensor) -> torch.Tensor:
        """将常见范围的张量稳健地映射/截断到 [-1,1]。"""
        zmin = float(z.min())
        zmax = float(z.max())
        if zmin >= 0.0 and zmax <= 1.0:
            # [0,1] → [-1,1]
            return z * 2.0 - 1.0
        if zmin >= -1.05 and zmax <= 1.05:
            # 近似已在 [-1,1]
            return z.clamp(-1.0, 1.0)
        if zmin >= 0.0 and zmax <= 255.0:
            # [0,255] → [-1,1]
            return z / 127.5 - 1.0
        # 兜底：直接截断，避免报错（最好从上游约束网络输出范围）
        return z.clamp(-1.0, 1.0)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # 1) 单通道 → 三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        # 2) 统一到 [-1,1]
        x = self._to_neg1to1(x)
        y = self._to_neg1to1(y)

        # 3) 小图插值到 >=64（保持长宽比为目标大小）
        H, W = x.shape[-2], x.shape[-1]
        new_h = max(H, self.min_size)
        new_w = max(W, self.min_size)
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(new_h, new_w), mode="bilinear", align_corners=False)

        if self.use_lpips:
            return self.loss(x.to(self.device), y.to(self.device)).mean().item()
        raise RuntimeError(
            "LPIPS requires torchmetrics. Install it with: pip install -e '.[metrics]'"
        )
