from __future__ import annotations

from pathlib import Path
from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def build_torch_route_aggregation_model(input_dim: int = 12):
    import torch.nn as nn

    class TinyRouteAggregationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_encoder = nn.Sequential(nn.Linear(input_dim, 24), nn.ReLU(), nn.Linear(24, 12), nn.ReLU())
            self.score_head = nn.Linear(12, 1)
            self.review_head = nn.Linear(12, 3)

        def forward(self, x):
            z = self.node_encoder(x)
            return {"score": self.score_head(z), "review_logits": self.review_head(z)}

    return TinyRouteAggregationNet()


def save_initialized_route_aggregation_model(path: str | Path) -> dict[str, Any]:
    import torch

    model = build_torch_route_aggregation_model()
    torch.save({"state_dict": model.state_dict(), "trained_steps": 0, "graph_schema": "module12_route_graph_v1"}, path)
    return {"torch_available": True, "trained_steps": 0, "graph_schema": "module12_route_graph_v1"}


def load_torch_route_aggregation_model(weights_path: str | Path):
    import torch

    model = build_torch_route_aggregation_model()
    payload = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
