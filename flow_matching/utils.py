import random

import numpy as np
import torch
from torch import Tensor, nn


def expand_t_like_x(t: float | Tensor, x: Tensor) -> Tensor:
    """Expand time vector t to match the shape of tensor x without making a copy.

    Args:
        t (float | Tensor): 1d tensor with shape (batch_size,).
        x (Tensor): Any tensor with shape (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...)

    Examples:
        >>> expand_t_like_x(0.5, torch.randn(10, 1, 28, 28)) # 0.5
        >>> expand_t_like_x(torch.rand(10), torch.randn(10, 1, 28, 28)) # (10, 1, 28, 28)
    """

    if isinstance(t, float):
        # We do not expand scalar values
        return t

    assert t.ndim == 1, "Time vector t must be a 1d tensor with shape (batch_size,)."
    assert t.size(0) == x.size(0), "Time vector t must have the same batch size as x."

    return t.reshape(-1, *([1] * (x.ndim - 1))).expand_as(x)


def model_size_summary(model: nn.Module, verbose: bool = True) -> str:
    """
    Return model summary as a string. Optionally print it to standard output.

    Args:
        model (nn.Module): PyTorch model to summarize.
        verbose (bool): If True, print the summary to standard output. Defaults to True.

    Returns:
        str: The model summary as a string.
    """

    def count_params(module: nn.Module, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))

    total_params = count_params(model, trainable_only=False)
    trainable_params = count_params(model, trainable_only=True)

    named_children = list(model.named_children())
    if not named_children:
        named_children = [("No submodules", None)]

    name_max_len = max(len(n) for n, _ in named_children)
    idx_width = len(str(len(named_children)))
    header = f'{"Idx".ljust(idx_width)} | {"Name".ljust(name_max_len)} | Params (ratio %)'

    lines = []
    lines.append(f"Model summary: {getattr(model, 'name_or_path', '')}")
    lines.append(header)
    lines.append("-" * len(header))

    for i, (child_name, child_module) in enumerate(named_children):
        if child_module is None:
            child_params = 0
        else:
            child_params = count_params(child_module, trainable_only=False)

        ratio = (child_params / total_params * 100) if total_params else 0
        lines.append(
            f"{str(i).ljust(idx_width)} | " f"{child_name.ljust(name_max_len)} | " f"{child_params:>11,} ({ratio:.2f}%)"
        )

    lines.append("-" * len(header))
    lines.append(f"Trainable params     : {trainable_params:,}")
    lines.append(f"Non-trainable params : {total_params - trainable_params:,}")
    lines.append(f"Total params         : {total_params:,}")

    # memory footprint
    if hasattr(model, "get_memory_footprint"):
        size_in_bytes = model.get_memory_footprint()
        device = getattr(model, "device", "unknown")
        lines.append(f"Memory footprint     : {size_in_bytes / 10**6:,.2f} MB (device={device})")
    else:
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        size_in_mb = (param_size + buffer_size) / 10**6
        device = getattr(model, "device", next(model.parameters()).device)
        lines.append(f"Memory footprint     : {size_in_mb:,.2f} MB (device={device})")

    dtype = getattr(model, "dtype", next(model.parameters()).dtype)
    lines.append(f"Model dtype          : {dtype}")

    summary = "\n".join(lines)

    if verbose:
        print(summary)

    return summary


def set_seed(seed: int) -> None:
    """Set the seed for the random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
