from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.fusion.conformal import ConformalCalibrationConfig, ConformalCalibrator
from rbccps_measurement.fusion.torch_model import save_initialized_conformal_model, torch_available


@dataclass(frozen=True)
class ConformalTrainingResult:
    checkpoint_json: Path
    checkpoint_weights: Path | None
    tracks_seen: int
    calibration_rows: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "checkpoint_weights": str(self.checkpoint_weights) if self.checkpoint_weights else None,
            "tracks_seen": self.tracks_seen,
            "calibration_rows": self.calibration_rows,
            "status": self.status,
            "summary": self.summary,
        }


def train_conformal_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> ConformalTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    tracks_seen = _count_lamp_tracks(root)
    calibration_rows = len(_read_rows(root / "annotations" / "qa_flags.csv")) + len(_read_rows(root / "annotations" / "attribution_labels.csv"))
    weights_path: Path | None = None
    summary: dict[str, Any] = {
        "module": "conformal",
        "architecture": "group_conditioned_conformal_abstention_v1",
        "tracks_seen": tracks_seen,
        "calibration_rows": calibration_rows,
        "group_key_schema": ["device_id", "route_group", "capture_mode", "confounder_density_bin", "gps_quality", "hdr_night_state"],
        "losses_planned": ["grouped_nll_brier", "risk_sensitive_abstention", "conformal_coverage_regularizer", "validity_regularizer"],
        "note": "Initialized calibration profile; empirical conformal thresholds require held-out labeled calibration data.",
    }
    status = "initialized_no_torch"
    if torch_available():
        weights_path = output / "conformal_checkpoint.pt"
        summary.update(save_initialized_conformal_model(weights_path))
        status = "initialized_no_calibration_labels"
    else:
        summary["torch_available"] = False
    checkpoint = {
        "checkpoint_type": "group_conditioned_conformal_checkpoint",
        "module": "conformal",
        "status": status,
        "weights": weights_path.name if weights_path else None,
        "config": config.to_dict(),
        "label_maps": {"actions": ["report", "abstain"], "ordinal": ["unknown", "poor", "marginal", "adequate"]},
        "training_summary": summary,
        "fallback": "deterministic_group_conformal_abstention_v1",
    }
    checkpoint_json = output / "conformal_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    ConformalCalibrator.from_checkpoint(checkpoint_json)
    return ConformalTrainingResult(checkpoint_json, weights_path, tracks_seen, calibration_rows, status, summary)


def _load_config(config_path: str | Path | None) -> ConformalCalibrationConfig:
    if not config_path:
        return ConformalCalibrationConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return ConformalCalibrationConfig(**{key: payload.get(key, value) for key, value in ConformalCalibrationConfig().__dict__.items()})


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
