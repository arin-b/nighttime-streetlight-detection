from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.route.graph_aggregation import EDGE_TYPES, IMPLEMENTATION, ORDINAL_CLASSES, RouteAggregationConfig
from rbccps_measurement.route.torch_model import save_initialized_route_aggregation_model, torch_available
from rbccps_measurement.training.readiness import _load_prepared_dataset


@dataclass(frozen=True)
class RouteAggregationTrainingResult:
    checkpoint_json: Path
    weights_path: Path | None
    clips_seen: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "clips_seen": self.clips_seen,
            "status": self.status,
            "summary": self.summary,
        }


def train_route_aggregation_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> RouteAggregationTrainingResult:
    root = Path(dataset_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config_payload = _load_config(config_path)
    config = RouteAggregationConfig(**{key: config_payload.get(key, value) for key, value in RouteAggregationConfig().__dict__.items()})
    try:
        clip_paths = _load_prepared_dataset(root)
    except Exception:
        clip_paths = []

    weights_path: Path | None = None
    torch_summary: dict[str, Any] = {"torch_available": False, "trained_steps": 0}
    if torch_available():
        weights_path = out / "route_aggregation_checkpoint.pt"
        torch_summary = save_initialized_route_aggregation_model(weights_path)

    status = "initialized_not_optimized"
    summary = {
        "clips_seen": len(clip_paths),
        "matching_policy": ["mapped_lamp_id", "lamp_inventory_id", "gps_proximity", "observation_identity_fallback"],
        "edge_types": list(EDGE_TYPES),
        "ordinal_classes": list(ORDINAL_CLASSES),
        "torch": torch_summary,
        "note": "Initialized route graph aggregation checkpoint; supervised optimization requires repeated-pass or inventory-linked labels.",
    }
    checkpoint = {
        "checkpoint_type": "module12_route_graph_aggregation_checkpoint",
        "module": "route_aggregation",
        "implementation": IMPLEMENTATION,
        "dataset": str(root),
        "config": config.to_dict(),
        "weights": weights_path.name if weights_path else None,
        "status": status,
        "graph_schema": {
            "node_types": ["observation", "candidate_lamp", "drive_pass", "gps_neighborhood", "road_segment", "map_prior"],
            "edge_types": list(EDGE_TYPES),
        },
        "losses_planned": ["candidate_link_loss", "ordinal_consensus_loss", "manual_review_priority_loss", "segment_underlighting_loss"],
        "fallback": IMPLEMENTATION,
        "training_summary": summary,
    }
    checkpoint_json = out / "route_aggregation_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    return RouteAggregationTrainingResult(checkpoint_json, weights_path, len(clip_paths), status, summary)


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    return json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
