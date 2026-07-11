from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.geometry.lamp_footprint_field import FootprintConfig, FootprintEstimator


@dataclass(frozen=True)
class FootprintTrainingResult:
    checkpoint_json: Path
    tracks_seen: int
    affected_rows: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "tracks_seen": self.tracks_seen,
            "affected_rows": self.affected_rows,
            "status": self.status,
            "summary": self.summary,
        }


def train_footprint_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> FootprintTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    tracks_seen = _count_lamp_tracks(root)
    affected_rows = _read_rows(root / "annotations" / "affected_regions.csv")
    region_histogram: dict[str, int] = {}
    for row in affected_rows:
        label = str(row.get("region_type") or row.get("normalized_label") or "").strip()
        if label:
            region_histogram[label] = region_histogram.get(label, 0) + 1

    status = "initialized_no_affected_region_labels" if not affected_rows else "trained_region_priors"
    summary = {
        "module": "footprint",
        "architecture": "lamp_query_public_space_cross_attention_field_v1",
        "tracks_seen": tracks_seen,
        "affected_rows": len(affected_rows),
        "region_histogram": region_histogram,
        "losses_planned": [
            "affected_field_binary_cross_entropy",
            "public_space_support_constraint",
            "occlusion_gate_consistency",
            "geometry_quality_calibration",
            "weak_distance_decay_regularizer",
        ],
        "note": "Dense optimization starts once affected-region masks are available; current checkpoint preserves deterministic field constraints and priors.",
    }
    checkpoint = {
        "checkpoint_type": "lamp_conditioned_affected_region_checkpoint",
        "module": "footprint",
        "status": status,
        "config": config.to_dict(),
        "label_maps": {
            "affected_region": ["affected_road", "affected_footpath", "affected_crossing", "affected_verge", "lit_area", "unknown"],
            "region_mix": ["road", "footpath", "crossing", "verge"],
        },
        "training_summary": summary,
        "fallback": "deterministic_lamp_conditioned_field_v1",
    }
    checkpoint_json = output / "footprint_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    FootprintEstimator.from_checkpoint(checkpoint_json)
    return FootprintTrainingResult(checkpoint_json, tracks_seen, len(affected_rows), status, summary)


def _load_config(config_path: str | Path | None) -> FootprintConfig:
    if not config_path:
        return FootprintConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return FootprintConfig(
        implementation=str(payload.get("implementation", "deterministic_lamp_conditioned_field_v1")),
        distance_decay_fraction=float(payload.get("distance_decay_fraction", 0.38)),
        downward_gate_strength=float(payload.get("downward_gate_strength", 12.0)),
        occlusion_suppression=float(payload.get("occlusion_suppression", 0.78)),
        weak_geometry_floor=float(payload.get("weak_geometry_floor", 0.48)),
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
        if not clip_path.exists():
            continue
        clip = ClipManifest.load(clip_path)
        total += len({track.track_id for track in clip.tracks if track.class_name == "streetlight_lamp_head"})
    return total


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
