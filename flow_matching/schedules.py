# file: flow_matching/schedules.py
import torch


def sigma(t, sigma_max, kind="sqrt"):
    t = torch.as_tensor(t, dtype=torch.float32, device=(t.device if torch.is_tensor(t) else None))
    if kind == "sqrt":   # σ(t) = σ_max * sqrt(t)
        return sigma_max * torch.sqrt(torch.clamp(t, min=1e-8))
    elif kind == "linear":  # σ(t) = σ_max * t
        return sigma_max * t
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")

def dsigma_dt(t, sigma_max, kind="sqrt"):
    t = torch.as_tensor(t, dtype=torch.float32, device=(t.device if torch.is_tensor(t) else None))
    if kind == "sqrt":   # dσ/dt = 0.5 σ_max / sqrt(t)
        return 0.5 * sigma_max / torch.sqrt(torch.clamp(t, min=1e-8))
    elif kind == "linear":  # dσ/dt = σ_max
        return torch.full_like(t, fill_value=float(sigma_max))
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")

def t_from_sigma_ch(sigma_ch, sigma_max, kind="sqrt"):
    if kind == "sqrt":
        return (sigma_ch / sigma_max) ** 2
    elif kind == "linear":
        return (sigma_ch / sigma_max)
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")
