from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.decomposition.source_slots import SOURCE_CLASSES, SourceDecompositionConfig, SourceDecompositionEstimator
from rbccps_measurement.decomposition.torch_models import save_initialized_source_model, torch_available


@dataclass(frozen=True)
class SourceDecompositionTrainingResult:
    checkpoint_json: Path
    checkpoint_weights: Path | None
    tracks_seen: int
    confounder_rows: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "checkpoint_weights": str(self.checkpoint_weights) if self.checkpoint_weights else None,
            "tracks_seen": self.tracks_seen,
            "confounder_rows": self.confounder_rows,
            "status": self.status,
            "summary": self.summary,
        }


def train_source_decomposition_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> SourceDecompositionTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    tracks_seen = _count_lamp_tracks(root)
    confounder_rows = _read_rows(root / "annotations" / "confounders.csv")
    histogram: dict[str, int] = {}
    for row in confounder_rows:
        label = str(row.get("confounder_type") or row.get("source_type") or row.get("normalized_label") or "").strip()
        if label:
            histogram[label] = histogram.get(label, 0) + 1

    weights_path: Path | None = None
    summary: dict[str, Any] = {
        "module": "source_decomposition",
        "architecture": "cnn_slot_decoder_source_classifier_v1",
        "tracks_seen": tracks_seen,
        "confounder_rows": len(confounder_rows),
        "confounder_histogram": histogram,
        "losses_planned": [
            "additive_reconstruction_loss",
            "source_classification_loss",
            "temporal_consistency_loss",
            "slot_assignment_loss",
            "sparsity_disentanglement_loss",
        ],
        "note": "Dense source-field labels are not required for initialization; deterministic fallback remains active until optimized weights are available.",
    }
    status = "initialized_no_torch"
    if torch_available():
        weights_path = output / "source_decomposition_checkpoint.pt"
        summary.update(save_initialized_source_model(weights_path, SOURCE_CLASSES))
        status = "initialized_no_dense_source_labels"
    else:
        summary["torch_available"] = False

    checkpoint = {
        "checkpoint_type": "source_decomposition_checkpoint",
        "module": "source_decomposition",
        "status": status,
        "weights": weights_path.name if weights_path else None,
        "config": config.to_dict(),
        "label_maps": {"source": list(SOURCE_CLASSES)},
        "training_summary": summary,
        "fallback": "deterministic_source_slots_v1",
    }
    checkpoint_json = output / "source_decomposition_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    SourceDecompositionEstimator.from_checkpoint(checkpoint_json)
    return SourceDecompositionTrainingResult(checkpoint_json, weights_path, tracks_seen, len(confounder_rows), status, summary)


def _load_config(config_path: str | Path | None) -> SourceDecompositionConfig:
    if not config_path:
        return SourceDecompositionConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return SourceDecompositionConfig(
        implementation=str(payload.get("implementation", "deterministic_source_slots_v1")),
        target_prior_weight=float(payload.get("target_prior_weight", 1.25)),
        other_lamp_prior_weight=float(payload.get("other_lamp_prior_weight", 0.35)),
        headlight_prior_weight=float(payload.get("headlight_prior_weight", 0.85)),
        shopfront_prior_weight=float(payload.get("shopfront_prior_weight", 0.75)),
        signal_prior_weight=float(payload.get("signal_prior_weight", 0.70)),
        reflection_prior_weight=float(payload.get("reflection_prior_weight", 0.80)),
        unknown_prior_weight=float(payload.get("unknown_prior_weight", 0.28)),
        residual_tolerance=float(payload.get("residual_tolerance", 0.18)),
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
