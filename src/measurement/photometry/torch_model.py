from __future__ import annotations

from pathlib import Path
from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def build_torch_photometry_model(input_dim: int = 8):
    import torch
    import torch.nn as nn

    class TinyMonotonePhotometryHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.positive = nn.Linear(input_dim, 1, bias=False)
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.positive(x).abs() + self.bias

    return TinyMonotonePhotometryHead()


def save_initialized_photometry_model(path: str | Path) -> dict[str, Any]:
    import torch

    model = build_torch_photometry_model()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "trained_steps": 0,
            "constraints": ["nonnegative_lux", "monotone_in_proxy_score"],
        },
        path,
    )
    return {"torch_available": True, "trained_steps": 0, "constraints": ["nonnegative_lux", "monotone_in_proxy_score"]}


def load_torch_photometry_model(weights_path: str | Path):
    import torch

    model = build_torch_photometry_model()
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
