from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.features.distributional_coverage import DistributionalFeatureConfig, DistributionalFeatureEstimator
from rbccps_measurement.features.torch_model import save_initialized_feature_model, torch_available


@dataclass(frozen=True)
class FeatureTrainingResult:
    checkpoint_json: Path
    checkpoint_weights: Path | None
    frames_seen: int
    affected_rows: int
    visibility_rows: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "checkpoint_weights": str(self.checkpoint_weights) if self.checkpoint_weights else None,
            "frames_seen": self.frames_seen,
            "affected_rows": self.affected_rows,
            "visibility_rows": self.visibility_rows,
            "status": self.status,
            "summary": self.summary,
        }


def train_features_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> FeatureTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    frames_seen = _count_frames(root)
    affected_rows = len(_read_rows(root / "annotations" / "affected_regions.csv"))
    visibility_rows = len(_read_rows(root / "annotations" / "visibility_labels.csv"))
    weights_path: Path | None = None
    summary: dict[str, Any] = {
        "module": "features",
        "architecture": "region_conditioned_distributional_feature_heads_v1",
        "frames_seen": frames_seen,
        "affected_rows": affected_rows,
        "visibility_rows": visibility_rows,
        "losses_planned": ["quantile_pinball_loss", "ordinal_adequacy_loss", "dark_hole_bce_iou", "temporal_consistency_loss"],
        "note": "Initialized distributional feature model; deterministic head remains active until dense labels are available.",
    }
    status = "initialized_no_torch"
    if torch_available():
        weights_path = output / "features_checkpoint.pt"
        summary.update(save_initialized_feature_model(weights_path))
        status = "initialized_no_distributional_labels"
    else:
        summary["torch_available"] = False

    checkpoint = {
        "checkpoint_type": "distributional_features_checkpoint",
        "module": "features",
        "status": status,
        "weights": weights_path.name if weights_path else None,
        "config": config.to_dict(),
        "label_maps": {
            "quantiles": ["q10", "q25", "q50", "q75", "q90"],
            "adequacy": ["unknown", "poor", "marginal", "adequate"],
        },
        "training_summary": summary,
        "fallback": "deterministic_distributional_features_v1",
    }
    checkpoint_json = output / "features_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    DistributionalFeatureEstimator.from_checkpoint(checkpoint_json)
    return FeatureTrainingResult(checkpoint_json, weights_path, frames_seen, affected_rows, visibility_rows, status, summary)


def _load_config(config_path: str | Path | None) -> DistributionalFeatureConfig:
    if not config_path:
        return DistributionalFeatureConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return DistributionalFeatureConfig(
        implementation=str(payload.get("implementation", "deterministic_distributional_features_v1")),
        base_useful_threshold=float(payload.get("base_useful_threshold", 0.34)),
        dark_hole_ratio=float(payload.get("dark_hole_ratio", 0.70)),
        min_region_mass=float(payload.get("min_region_mass", 1e-6)),
    )


def _count_frames(root: Path) -> int:
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.exists():
        return 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    total = 0
    for item in payload.get("clips", []):
        clip_ref = Path(item["manifest"])
        clip_path = clip_ref if clip_ref.is_absolute() else root / clip_ref
        if clip_path.exists():
            total += len(ClipManifest.load(clip_path).frames)
    return total


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
