from __future__ import annotations

from pathlib import Path
from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def build_torch_fusion_model(input_dim: int = 16):
    import torch
    import torch.nn as nn

    class TinyMonotonicFusionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.positive = nn.Linear(5, 1, bias=False)
            self.negative = nn.Linear(6, 1, bias=False)
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, positive, negative):
            return self.positive(positive).abs() - self.negative(negative).abs() + self.bias

    return TinyMonotonicFusionNet()


def build_torch_conformal_model(input_dim: int = 10):
    import torch.nn as nn

    class TinyConformalRiskNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 3))

        def forward(self, x):
            return self.net(x)

    return TinyConformalRiskNet()


def save_initialized_fusion_model(path: str | Path) -> dict[str, Any]:
    import torch

    model = build_torch_fusion_model()
    torch.save({"state_dict": model.state_dict(), "trained_steps": 0, "constraints": "signed_monotonic"}, path)
    return {"torch_available": True, "trained_steps": 0, "constraints": "signed_monotonic"}


def save_initialized_conformal_model(path: str | Path) -> dict[str, Any]:
    import torch

    model = build_torch_conformal_model()
    torch.save({"state_dict": model.state_dict(), "trained_steps": 0}, path)
    return {"torch_available": True, "trained_steps": 0}


def load_torch_fusion_model(weights_path: str | Path, checkpoint: dict[str, Any]):
    import torch

    model = build_torch_fusion_model()
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_torch_conformal_model(weights_path: str | Path, checkpoint: dict[str, Any]):
    import torch

    model = build_torch_conformal_model()
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
