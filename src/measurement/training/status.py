from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.status.crop_sequence import CropSequenceConfig, build_lamp_crop_sequence
from rbccps_measurement.status.model import STATUS_CLASSES, StatusEstimator, StatusModelConfig
from rbccps_measurement.status.torch_model import save_initialized_or_trained_model, torch_available


@dataclass(frozen=True)
class StatusTrainingResult:
    checkpoint_json: Path
    checkpoint_weights: Path | None
    samples_seen: int
    samples_used: int
    status: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_json": str(self.checkpoint_json),
            "checkpoint_weights": str(self.checkpoint_weights) if self.checkpoint_weights else None,
            "samples_seen": self.samples_seen,
            "samples_used": self.samples_used,
            "status": self.status,
            "summary": self.summary,
        }


def train_status_module(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> StatusTrainingResult:
    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    crop_config = _load_crop_config(config_path)
    model_config = StatusModelConfig(sequence_length=crop_config.sequence_length, crop_size=crop_config.crop_size)
    labels = _load_status_labels(root / "annotations" / "lamp_status.csv")
    samples = []
    samples_seen = 0

    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for clip_entry in dataset_manifest.get("clips", []):
            clip_ref = Path(clip_entry["manifest"])
            clip_path = clip_ref if clip_ref.is_absolute() else root / clip_ref
            if not clip_path.exists():
                continue
            clip = ClipManifest.load(clip_path)
            frames = clip.frame_by_id()
            frame_root = clip_path.parent
            by_track: dict[str, list] = {}
            for track in clip.tracks:
                if track.class_name != "streetlight_lamp_head":
                    continue
                by_track.setdefault(track.track_id, []).append(track)
            for track_id, records in by_track.items():
                samples_seen += 1
                label = labels.get((clip.clip_id, track_id))
                if not label:
                    continue
                sequence = build_lamp_crop_sequence(track_id, records, frames, frame_root, config=crop_config)
                if "image_missing" in sequence.quality_flags:
                    continue
                samples.append((sequence, label))

    weights_path: Path | None = None
    checkpoint_status = "initialized_no_torch"
    training_summary: dict[str, Any] = {
        "module": "status",
        "architecture": "cnn_gru_multi_head_v1",
        "status_classes": list(STATUS_CLASSES),
        "sequence_length": crop_config.sequence_length,
        "crop_size": crop_config.crop_size,
        "samples_seen": samples_seen,
        "samples_used": len(samples),
        "losses_planned": ["status_cross_entropy", "temporal_smoothness", "flicker_auxiliary", "exposure_invariance", "calibration_brier"],
    }
    if torch_available() and samples:
        weights_path = output / "status_checkpoint.pt"
        fit_summary = save_initialized_or_trained_model(weights_path, samples, model_config)
        checkpoint_status = "trained" if fit_summary.get("trained_steps", 0) else "initialized_no_optimized_steps"
        training_summary.update(fit_summary)
    elif torch_available():
        weights_path = output / "status_checkpoint.pt"
        fit_summary = save_initialized_or_trained_model(weights_path, [], model_config)
        checkpoint_status = "initialized_no_labeled_samples"
        training_summary.update(fit_summary)
    else:
        training_summary["torch_available"] = False

    checkpoint = {
        "checkpoint_type": "latent_emission_state_checkpoint",
        "module": "status",
        "status": checkpoint_status,
        "weights": weights_path.name if weights_path else None,
        "model_config": model_config.to_dict(),
        "crop_config": {
            "sequence_length": crop_config.sequence_length,
            "crop_size": crop_config.crop_size,
            "padding_ratio": crop_config.padding_ratio,
        },
        "label_maps": {
            "status": list(STATUS_CLASSES),
            "emission": ["off", "dim", "on"],
            "occlusion": ["clear", "partial", "occluded"],
            "capture": ["clean", "saturated", "blur_or_unreliable", "unknown"],
            "flicker": ["stable", "possible_flicker", "flicker"],
        },
        "training_summary": training_summary,
        "fallback": "deterministic_latent_status_v1",
    }
    checkpoint_json = output / "status_checkpoint.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    StatusEstimator.from_checkpoint(checkpoint_json)
    return StatusTrainingResult(
        checkpoint_json=checkpoint_json,
        checkpoint_weights=weights_path,
        samples_seen=samples_seen,
        samples_used=len(samples),
        status=checkpoint_status,
        summary=training_summary,
    )


def _load_crop_config(config_path: str | Path | None) -> CropSequenceConfig:
    if not config_path:
        return CropSequenceConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return CropSequenceConfig(
        sequence_length=int(payload.get("sequence_length", 16)),
        crop_size=int(payload.get("crop_size", 64)),
        padding_ratio=float(payload.get("padding_ratio", 0.35)),
    )


def _load_status_labels(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    labels: dict[tuple[str, str], str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("status") or row.get("normalized_label") or "").strip()
            clip_id = str(row.get("clip_id") or "").strip()
            track_id = str(row.get("track_id") or "").strip()
            if clip_id and track_id and label in STATUS_CLASSES:
                labels[(clip_id, track_id)] = label
    return labels
