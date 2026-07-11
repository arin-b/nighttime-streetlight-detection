from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.attribution.counterfactual import CounterfactualAttributionConfig, CounterfactualAttributionEstimator
from rbccps_measurement.attribution.torch_model import save_initialized_attribution_model, torch_available
from rbccps_measurement.contracts.input_schema import ClipManifest


@dataclass(frozen=True)
class AttributionTrainingResult:
    checkpoint_json: Path
    checkpoint_weights: Path | None
    tracks_seen: int
    attribution_rows: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "checkpoint_weights": str(self.checkpoint_weights) if self.checkpoint_weights else None,
            "tracks_seen": self.tracks_seen,
            "attribution_rows": self.attribution_rows,
            "status": self.status,
            "summary": self.summary,
        }


def train_attribution_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> AttributionTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    tracks_seen = _count_lamp_tracks(root)
    rows = _read_rows(root / "annotations" / "attribution_labels.csv")
    histogram: dict[str, int] = {}
    for row in rows:
        label = str(row.get("attribution_class") or row.get("normalized_label") or "").strip()
        if label:
            histogram[label] = histogram.get(label, 0) + 1
    weights_path: Path | None = None
    summary: dict[str, Any] = {
        "module": "attribution",
        "architecture": "counterfactual_utility_difference_head_v1",
        "tracks_seen": tracks_seen,
        "attribution_rows": len(rows),
        "attribution_histogram": histogram,
        "losses_planned": ["attribution_classification_loss", "counterfactual_consistency_loss", "mixed_source_ranking_loss"],
        "note": "Initialized attribution model; deterministic counterfactual utility remains active until labels are available.",
    }
    status = "initialized_no_torch"
    if torch_available():
        weights_path = output / "attribution_checkpoint.pt"
        summary.update(save_initialized_attribution_model(weights_path))
        status = "initialized_no_counterfactual_labels"
    else:
        summary["torch_available"] = False

    checkpoint = {
        "checkpoint_type": "counterfactual_attribution_checkpoint",
        "module": "attribution",
        "status": status,
        "weights": weights_path.name if weights_path else None,
        "config": config.to_dict(),
        "label_maps": {"attribution": ["certain", "mixed", "uncertain"]},
        "training_summary": summary,
        "fallback": "deterministic_counterfactual_attribution_v1",
    }
    checkpoint_json = output / "attribution_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    CounterfactualAttributionEstimator.from_checkpoint(checkpoint_json)
    return AttributionTrainingResult(checkpoint_json, weights_path, tracks_seen, len(rows), status, summary)


def _load_config(config_path: str | Path | None) -> CounterfactualAttributionConfig:
    if not config_path:
        return CounterfactualAttributionConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return CounterfactualAttributionConfig(
        implementation=str(payload.get("implementation", "deterministic_counterfactual_attribution_v1")),
        useful_threshold=float(payload.get("useful_threshold", 0.34)),
        mixed_competition_threshold=float(payload.get("mixed_competition_threshold", 0.35)),
        certain_score_threshold=float(payload.get("certain_score_threshold", 0.55)),
    )


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


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
