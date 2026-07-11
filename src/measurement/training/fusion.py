from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.fusion.monotonic_heads import EDGE_TYPES, ORDINAL_CLASSES, MonotonicFusionConfig, MonotonicFusionEstimator
from rbccps_measurement.fusion.torch_model import save_initialized_fusion_model, torch_available


@dataclass(frozen=True)
class FusionTrainingResult:
    checkpoint_json: Path
    checkpoint_weights: Path | None
    tracks_seen: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "checkpoint_weights": str(self.checkpoint_weights) if self.checkpoint_weights else None,
            "tracks_seen": self.tracks_seen,
            "status": self.status,
            "summary": self.summary,
        }


def train_fusion_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> FusionTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    tracks_seen = _count_lamp_tracks(root)
    weights_path: Path | None = None
    summary: dict[str, Any] = {
        "module": "fusion",
        "architecture": "monotonic_scene_graph_fusion_v1",
        "tracks_seen": tracks_seen,
        "edge_types": list(EDGE_TYPES),
        "ordinal_classes": list(ORDINAL_CLASSES),
        "losses_planned": ["ordinal_classification_loss", "monotonicity_penalty", "confidence_calibration_loss"],
        "note": "Initialized monotonic fusion checkpoint; deterministic constrained head remains active until supervised labels exist.",
    }
    status = "initialized_no_torch"
    if torch_available():
        weights_path = output / "fusion_checkpoint.pt"
        summary.update(save_initialized_fusion_model(weights_path))
        status = "initialized_no_fusion_labels"
    else:
        summary["torch_available"] = False
    checkpoint = {
        "checkpoint_type": "monotonic_scene_graph_fusion_checkpoint",
        "module": "fusion",
        "status": status,
        "weights": weights_path.name if weights_path else None,
        "config": config.to_dict(),
        "label_maps": {"ordinal": list(ORDINAL_CLASSES), "edge_types": list(EDGE_TYPES)},
        "monotonic_constraints": {
            "positive": ["coverage", "adequacy", "uniformity", "temporal_stability", "attribution"],
            "negative": ["glare", "confounder", "occlusion", "dark_hole", "attribution_uncertainty", "missingness"],
        },
        "training_summary": summary,
        "fallback": "deterministic_monotonic_scene_graph_fusion_v1",
    }
    checkpoint_json = output / "fusion_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    MonotonicFusionEstimator.from_checkpoint(checkpoint_json)
    return FusionTrainingResult(checkpoint_json, weights_path, tracks_seen, status, summary)


def _load_config(config_path: str | Path | None) -> MonotonicFusionConfig:
    if not config_path:
        return MonotonicFusionConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return MonotonicFusionConfig(**{key: payload.get(key, value) for key, value in MonotonicFusionConfig().__dict__.items()})


def _count_lamp_tracks(root: Path) -> int:
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.exists():
        return 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    total = 0
    for item in payload.get("clips", []):
        clip_ref = Path(item["manifest"])
        clip_path = clip_ref if clip_ref.is_absolute() else root / clip_ref
        if clip_path.exists():
            clip = ClipManifest.load(clip_path)
            total += len({track.track_id for track in clip.tracks if track.class_name == "streetlight_lamp_head"})
    return total
