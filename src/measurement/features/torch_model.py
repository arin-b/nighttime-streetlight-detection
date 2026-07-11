from __future__ import annotations

from pathlib import Path
from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def build_torch_feature_model(input_channels: int = 6):
    import torch.nn as nn

    class TinyDistributionalFeatureNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 24, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.dark_hole = nn.Conv2d(24, 1, kernel_size=1)
            self.pooled = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(24, 14))

        def forward(self, x):
            features = self.encoder(x)
            pooled = self.pooled(features)
            return {"feature_vector": pooled, "dark_hole_logits": self.dark_hole(features)}

    return TinyDistributionalFeatureNet()


def save_initialized_feature_model(path: str | Path) -> dict[str, Any]:
    import torch

    model = build_torch_feature_model()
    torch.save({"state_dict": model.state_dict(), "trained_steps": 0}, path)
    return {"torch_available": True, "trained_steps": 0}


def load_torch_feature_model(weights_path: str | Path, checkpoint: dict[str, Any]):
    import torch

    model = build_torch_feature_model()
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
