from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.segmentation.illumination_disentangled import (
    CONFOUNDER_CLASSES,
    OCCLUDER_CLASSES,
    PUBLIC_SPACE_CLASSES,
    SEMANTIC_CLASSES,
    SegmentationConfig,
    SegmentationEstimator,
)


@dataclass(frozen=True)
class SegmentationTrainingResult:
    checkpoint_json: Path
    frames_seen: int
    annotation_rows: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "frames_seen": self.frames_seen,
            "annotation_rows": self.annotation_rows,
            "status": self.status,
            "summary": self.summary,
        }


def train_segmentation_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> SegmentationTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    frames_seen = _count_frames(root)
    public_rows = _read_rows(root / "annotations" / "public_space_regions.csv")
    confounder_rows = _read_rows(root / "annotations" / "confounders.csv")
    annotation_rows = len(public_rows) + len(confounder_rows)
    class_histogram: dict[str, int] = {}
    for row in public_rows:
        label = str(row.get("region_type") or row.get("normalized_label") or "").strip()
        if label:
            class_histogram[label] = class_histogram.get(label, 0) + 1
    for row in confounder_rows:
        label = str(row.get("confounder_type") or row.get("source_type") or row.get("normalized_label") or "").strip()
        if label:
            class_histogram[label] = class_histogram.get(label, 0) + 1

    status = "initialized_no_dense_masks" if annotation_rows == 0 else "trained_annotation_priors"
    summary = {
        "module": "segmentation",
        "architecture": "raw_aux_enhanced_cross_attention_multi_decoder_v1",
        "frames_seen": frames_seen,
        "annotation_rows": annotation_rows,
        "class_histogram": class_histogram,
        "losses_planned": [
            "segmentation_cross_entropy",
            "boundary_loss",
            "exposure_perturbation_consistency",
            "night_style_consistency",
            "uncertainty_loss",
            "confounder_separation_loss",
        ],
        "note": "Dense neural optimization is enabled once semantic mask rasters are collected; current checkpoint fixes label maps, priors, and deterministic fallback.",
    }
    checkpoint = {
        "checkpoint_type": "illumination_disentangled_segmentation_checkpoint",
        "module": "segmentation",
        "status": status,
        "config": config.to_dict(),
        "label_maps": {
            "semantic": list(SEMANTIC_CLASSES),
            "public_space": list(PUBLIC_SPACE_CLASSES),
            "occluder": list(OCCLUDER_CLASSES),
            "confounder": list(CONFOUNDER_CLASSES),
        },
        "training_summary": summary,
        "fallback": "deterministic_illumination_disentangled_v1",
    }
    checkpoint_json = output / "segmentation_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    SegmentationEstimator.from_checkpoint(checkpoint_json)
    return SegmentationTrainingResult(checkpoint_json, frames_seen, annotation_rows, status, summary)


def _load_config(config_path: str | Path | None) -> SegmentationConfig:
    if not config_path:
        return SegmentationConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return SegmentationConfig(
        implementation=str(payload.get("implementation", "deterministic_illumination_disentangled_v1")),
        uncertainty_floor=float(payload.get("uncertainty_floor", 0.08)),
        saturation_uncertainty_weight=float(payload.get("saturation_uncertainty_weight", 0.45)),
        glare_confounder_weight=float(payload.get("glare_confounder_weight", 0.65)),
        enhanced_view_policy=str(payload.get("enhanced_view_policy", "auxiliary_only")),
        measurement_source=str(payload.get("measurement_source", "original_or_normalized_capture")),
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
