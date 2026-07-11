from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.normalization.module1 import CaptureNormalizationConfig, load_rgb01


@dataclass(frozen=True)
class CaptureNormalizationTrainingResult:
    checkpoint_path: Path
    frames_seen: int
    frames_used: int
    skipped_frames: int
    config: CaptureNormalizationConfig
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "frames_seen": self.frames_seen,
            "frames_used": self.frames_used,
            "skipped_frames": self.skipped_frames,
            "config": self.config.to_dict(),
            "summary": self.summary,
        }


def train_capture_normalization(dataset_root: str | Path, out: str | Path, config_path: str | Path | None = None) -> CaptureNormalizationTrainingResult:
    """Fit the first deterministic Module-1 checkpoint from prepared frames.

    This is intentionally conservative: without field references or RAW/YUV
    calibration it estimates robust thresholds and metadata defaults, not a
    physical camera response curve.
    """

    root = Path(dataset_root)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8-sig"))
    base_config = _load_config(config_path)

    max_values: list[float] = []
    p95_values: list[float] = []
    exposure_times: list[float] = []
    isos: list[float] = []
    frames_seen = 0
    frames_used = 0
    skipped_frames = 0

    for clip_entry in dataset_manifest.get("clips", []):
        manifest_ref = Path(clip_entry["manifest"])
        manifest_path = manifest_ref if manifest_ref.is_absolute() else root / manifest_ref
        clip = ClipManifest.load(manifest_path)
        for frame in clip.frames:
            frames_seen += 1
            image_path = Path(frame.image_uri)
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            if not image_path.exists():
                skipped_frames += 1
                continue
            rgb = load_rgb01(image_path)
            max_channel = np.max(rgb, axis=2)
            max_values.append(float(np.quantile(max_channel, 0.999)))
            p95_values.append(float(np.quantile(max_channel, 0.95)))
            if frame.camera.exposure_time_s:
                exposure_times.append(float(frame.camera.exposure_time_s))
            if frame.camera.sensor_sensitivity_iso:
                isos.append(float(frame.camera.sensor_sensitivity_iso))
            frames_used += 1

    if frames_used:
        saturation_threshold = max(0.95, min(0.995, float(np.quantile(max_values, 0.95))))
        glare_threshold = max(0.85, min(0.98, float(np.quantile(max_values, 0.75))))
        bloom_threshold = max(0.72, min(0.95, float(np.quantile(p95_values, 0.90))))
        reference_exposure = float(np.median(exposure_times)) if exposure_times else base_config.reference_exposure_time_s
        reference_iso = float(np.median(isos)) if isos else base_config.reference_iso
    else:
        saturation_threshold = base_config.saturation_threshold
        glare_threshold = base_config.glare_threshold
        bloom_threshold = base_config.bloom_threshold
        reference_exposure = base_config.reference_exposure_time_s
        reference_iso = base_config.reference_iso

    trained_config = CaptureNormalizationConfig(
        implementation="deterministic_metadata_conditioned_v1",
        reference_exposure_time_s=reference_exposure,
        reference_iso=reference_iso,
        response_gamma=base_config.response_gamma,
        saturation_threshold=saturation_threshold,
        glare_threshold=glare_threshold,
        bloom_threshold=bloom_threshold,
        bloom_radius_px=base_config.bloom_radius_px,
        min_reliability=base_config.min_reliability,
    )
    summary = {
        "training_mode": "robust_threshold_fit_no_physical_claim",
        "frames_seen": frames_seen,
        "frames_used": frames_used,
        "skipped_frames": skipped_frames,
        "note": "No physical response curve is learned without calibration references; this checkpoint fixes deterministic thresholds and metadata defaults.",
    }
    checkpoint = {
        "checkpoint_type": "capture_normalization_checkpoint",
        "module": "normalization",
        "status": "trained_thresholds" if frames_used else "initialized_no_frames_available",
        "config": trained_config.to_dict(),
        "training_summary": summary,
    }
    checkpoint_path = output / "normalization_checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    return CaptureNormalizationTrainingResult(
        checkpoint_path=checkpoint_path,
        frames_seen=frames_seen,
        frames_used=frames_used,
        skipped_frames=skipped_frames,
        config=trained_config,
        summary=summary,
    )


def _load_config(config_path: str | Path | None) -> CaptureNormalizationConfig:
    if not config_path:
        return CaptureNormalizationConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return CaptureNormalizationConfig(
        implementation=str(payload.get("implementation", "deterministic_metadata_conditioned_v1")),
        reference_exposure_time_s=float(payload.get("reference_exposure_time_s", 0.0167)),
        reference_iso=float(payload.get("reference_iso", 800.0)),
        response_gamma=float(payload.get("response_gamma", 1.0)),
        saturation_threshold=float(payload.get("saturation_threshold", 0.98)),
        glare_threshold=float(payload.get("glare_threshold", 0.92)),
        bloom_threshold=float(payload.get("bloom_threshold", 0.86)),
        bloom_radius_px=int(payload.get("bloom_radius_px", 4)),
        min_reliability=float(payload.get("min_reliability", 0.05)),
    )
