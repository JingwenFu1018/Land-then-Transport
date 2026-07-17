"""Command-line configuration shared by the public training and evaluation entry points."""

from __future__ import annotations

import argparse
import os
import types
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints


def default_data_root() -> str:
    """Return the portable data root used when --data_root is omitted."""
    return os.environ.get("LTT_DATA_ROOT", str(Path("data")))


def parse_snr_list(value: str) -> list[float]:
    """Parse a comma-separated SNR list."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_dataclass(config_type):
    """Build a standard-library CLI parser from a configuration dataclass."""
    parser = argparse.ArgumentParser()
    type_hints = get_type_hints(config_type)
    for field in fields(config_type):
        annotation = type_hints[field.name]
        if get_origin(annotation) in (Union, types.UnionType):
            annotation = next(item for item in get_args(annotation) if item is not type(None))

        kwargs = {"default": field.default}
        if annotation is bool:
            kwargs.update(type=_parse_bool, nargs="?", const=True)
        else:
            kwargs["type"] = annotation
        parser.add_argument(f"--{field.name}", **kwargs)
    return config_type(**vars(parser.parse_args()))


@dataclass
class TrackingConfig:
    use_wandb: bool = False
    wandb_project: str = "land-then-transport"
    wandb_entity: str | None = None
    wandb_name: str | None = None


@dataclass
class TrainConfig(TrackingConfig):
    dataset: str = "mnist"
    data_root: str = default_data_root()
    download: bool = True
    output_dir: str = "outputs"
    batch_size: int = 128
    n_epochs: int = 100
    learning_rate: float = 1e-3
    horizontal_flip: bool = False
    seed: int = 42

    sigma_max: float | None = None
    sigma_schedule: str = "sqrt"
    min_snr_db: float = 0.0

    val_fraction: float = 0.1
    eval_every: int = 10
    max_val_batches: int = 200
    save_best: bool = True
    save_last: bool = False

    snr_db: float | None = None
    test_snr_db_list: str = ""
    ode_steps: int = 501

    channel: str = "awgn"
    csi_mode: str = "perfect"
    csi_noise_std: float = 0.0
    use_complex_channels: bool = False
    wandb_log_freq: int = 100


@dataclass
class EvaluateConfig(TrackingConfig):
    dataset: str = "mnist"
    data_root: str = default_data_root()
    download: bool = True
    output_dir: str = "outputs"
    batch_size: int = 128
    seed: int = 42

    checkpoint: str | None = None
    sigma_max: float | None = None
    sigma_schedule: str | None = None

    test_snr_db_list: str = "0,3,5,7,10,12,15"
    ode_steps: int = 501
    use_lpips: bool = True

    channel: str = "awgn"
    csi_mode: str = "perfect"
    csi_noise_std: float = 0.0
    use_complex_channels: bool = False
    mimo_corr_rho: float = 0.0
