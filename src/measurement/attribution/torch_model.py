from __future__ import annotations

from pathlib import Path
from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def build_torch_attribution_model(input_dim: int = 16):
    import torch.nn as nn

    class TinyCounterfactualAttributionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            self.score = nn.Linear(16, 1)
            self.klass = nn.Linear(16, 3)
            self.uncertainty = nn.Linear(16, 1)

        def forward(self, x):
            hidden = self.net(x)
            return {"score": self.score(hidden), "class_logits": self.klass(hidden), "uncertainty": self.uncertainty(hidden)}

    return TinyCounterfactualAttributionNet()


def save_initialized_attribution_model(path: str | Path) -> dict[str, Any]:
    import torch

    model = build_torch_attribution_model()
    torch.save({"state_dict": model.state_dict(), "trained_steps": 0, "classes": ["certain", "mixed", "uncertain"]}, path)
    return {"torch_available": True, "trained_steps": 0, "classes": ["certain", "mixed", "uncertain"]}


def load_torch_attribution_model(weights_path: str | Path, checkpoint: dict[str, Any]):
    import torch

    model = build_torch_attribution_model()
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
