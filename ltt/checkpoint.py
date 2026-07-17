"""Checkpoint naming, saving, discovery, and model reconstruction."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from flow_matching.models import UNetModel


def checkpoint_path(
    output_dir: Path,
    stem: str,
    epoch: int | None,
    sigma_max: float,
    learning_rate: float,
    schedule: str,
    channel: str,
) -> Path:
    epoch_label = "NA" if epoch is None else str(int(epoch))
    filename = (
        f"{stem}_e{epoch_label}-sched{schedule}-ch{channel}"
        f"-smax{float(sigma_max):.4f}-lr{float(learning_rate):.0e}.pth"
    )
    return output_dir / filename


def resolve_checkpoint(output_dir: Path, checkpoint: str | None) -> Path:
    """Resolve an explicit checkpoint or discover the newest local checkpoint."""
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    for pattern in ("ckpt_best*.pth", "ckpt_last*.pth", "ckpt*.pth"):
        candidates = list(output_dir.glob(pattern))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(
        f"No checkpoint found under {output_dir}. Pass --checkpoint explicitly."
    )


def save_checkpoint(
    path: Path,
    model: UNetModel,
    input_shape: tuple[int, ...],
    num_classes: int,
    sigma_max: float,
    sigma_schedule: str,
    epoch: int | None = None,
    val_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_args": {
            "input_shape": tuple(input_shape),
            "num_channels": 64,
            "num_res_blocks": 2,
            "num_classes": num_classes,
            "class_cond": True,
        },
        "sigma_max": float(sigma_max),
        "sigma_schedule": str(sigma_schedule),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(payload, path)
    print(f"[save] checkpoint -> {path}")


def _parse_float_from_name(path: Path, key: str) -> float | None:
    match = re.search(rf"{key}([0-9]*\.?[0-9]+)", path.name)
    return float(match.group(1)) if match else None


def _parse_schedule_from_name(path: Path) -> str | None:
    match = re.search(r"sched([A-Za-z0-9_]+)", path.name)
    return match.group(1).lower() if match else None


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    input_shape: tuple[int, ...] | None = None,
    num_classes: int | None = None,
) -> tuple[UNetModel, float | None, str]:
    """Load a trusted project checkpoint and rebuild its UNet."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "model_args" not in payload:
        raise RuntimeError(
            f"Invalid checkpoint format: {checkpoint_path}; expected state_dict and model_args."
        )

    model_args = payload["model_args"]
    resolved_shape = model_args.get("input_shape") or input_shape
    resolved_classes = model_args.get("num_classes") or num_classes
    if resolved_shape is None or resolved_classes is None:
        raise ValueError("Checkpoint does not contain enough information to rebuild the model.")

    model = UNetModel(
        resolved_shape,
        num_channels=model_args.get("num_channels", 64),
        num_res_blocks=model_args.get("num_res_blocks", 2),
        num_classes=resolved_classes,
        class_cond=model_args.get("class_cond", True),
    ).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing:
        print("[load] missing keys:", missing)
    if unexpected:
        print("[load] unexpected keys:", unexpected)

    sigma_max = payload.get("sigma_max")
    if sigma_max is None:
        sigma_max = _parse_float_from_name(checkpoint_path, "smax")
    schedule = payload.get("sigma_schedule") or _parse_schedule_from_name(checkpoint_path) or "sqrt"
    return model, sigma_max, schedule
