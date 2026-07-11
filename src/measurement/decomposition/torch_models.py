from __future__ import annotations

from pathlib import Path
from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def build_torch_source_model(num_sources: int = 7):
    import torch.nn as nn

    class TinySourceSlotNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 24, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.field_head = nn.Conv2d(24, num_sources, kernel_size=1)
            self.type_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(24, num_sources))

        def forward(self, x):
            features = self.encoder(x)
            return {"fields": self.field_head(features), "source_logits": self.type_head(features)}

    return TinySourceSlotNet()


def build_torch_ris_model():
    import torch.nn as nn

    class TinyRISNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 24, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.reflectance = nn.Conv2d(24, 3, kernel_size=1)
            self.illumination = nn.Conv2d(24, 1, kernel_size=1)
            self.source = nn.Conv2d(24, 1, kernel_size=1)
            self.confidence = nn.Conv2d(24, 1, kernel_size=1)

        def forward(self, x):
            features = self.encoder(x)
            return {
                "reflectance_like": self.reflectance(features),
                "illumination_like": self.illumination(features),
                "source_like": self.source(features),
                "confidence": self.confidence(features),
            }

    return TinyRISNet()


def save_initialized_source_model(path: str | Path, source_classes: tuple[str, ...]) -> dict[str, Any]:
    import torch

    model = build_torch_source_model(len(source_classes))
    torch.save({"state_dict": model.state_dict(), "source_classes": list(source_classes), "trained_steps": 0}, path)
    return {"torch_available": True, "trained_steps": 0, "source_classes": list(source_classes)}


def save_initialized_ris_model(path: str | Path) -> dict[str, Any]:
    import torch

    model = build_torch_ris_model()
    torch.save({"state_dict": model.state_dict(), "trained_steps": 0}, path)
    return {"torch_available": True, "trained_steps": 0}


def load_torch_source_model(weights_path: str | Path, checkpoint: dict[str, Any]):
    import torch

    classes = tuple(checkpoint.get("label_maps", {}).get("source", [])) or tuple(range(7))
    model = build_torch_source_model(len(classes))
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_torch_ris_model(weights_path: str | Path, checkpoint: dict[str, Any]):
    import torch

    model = build_torch_ris_model()
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
