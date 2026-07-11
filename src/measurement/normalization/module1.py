from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rbccps_measurement.contracts.input_schema import CameraMetadata, FrameRecord
from rbccps_measurement.contracts.module_io import NormalizedFrameProduct


@dataclass(frozen=True)
class CaptureNormalizationConfig:
    implementation: str = "deterministic_metadata_conditioned_v1"
    reference_exposure_time_s: float = 0.0167
    reference_iso: float = 800.0
    response_gamma: float = 1.0
    saturation_threshold: float = 0.98
    glare_threshold: float = 0.92
    bloom_threshold: float = 0.86
    bloom_radius_px: int = 4
    min_reliability: float = 0.05
    checkpoint_path: str | None = None
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "CaptureNormalizationConfig":
        checkpoint_path = Path(path)
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        config = payload.get("config", payload)
        return cls(
            implementation=str(config.get("implementation", "deterministic_metadata_conditioned_v1")),
            reference_exposure_time_s=float(config.get("reference_exposure_time_s", 0.0167)),
            reference_iso=float(config.get("reference_iso", 800.0)),
            response_gamma=float(config.get("response_gamma", 1.0)),
            saturation_threshold=float(config.get("saturation_threshold", 0.98)),
            glare_threshold=float(config.get("glare_threshold", 0.92)),
            bloom_threshold=float(config.get("bloom_threshold", 0.86)),
            bloom_radius_px=int(config.get("bloom_radius_px", 4)),
            min_reliability=float(config.get("min_reliability", 0.05)),
            checkpoint_path=str(checkpoint_path),
            checkpoint_metadata=payload.get("training_summary", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rgb01(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def srgb_inverse_luma(rgb01: np.ndarray) -> np.ndarray:
    linear = np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    return np.clip(0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2], 0.0, 1.0)


def exposure_factor(camera: CameraMetadata, config: CaptureNormalizationConfig | None = None) -> float:
    config = config or CaptureNormalizationConfig()
    exposure = camera.exposure_time_s or config.reference_exposure_time_s
    iso = camera.sensor_sensitivity_iso or config.reference_iso
    factor = (float(exposure) / max(1e-9, config.reference_exposure_time_s)) * (float(iso) / max(1e-9, config.reference_iso))
    return max(0.05, min(64.0, factor))


def _quality_penalty(camera: CameraMetadata) -> tuple[float, list[str]]:
    reliability = 1.0
    flags: list[str] = []
    if camera.metadata_quality in {"missing", "poor", "pseudo"}:
        flags.append("missing_or_poor_camera_metadata" if camera.metadata_quality != "pseudo" else "pseudo_camera_metadata")
        reliability -= 0.35 if camera.metadata_quality != "pseudo" else 0.25
    if camera.auto_exposure_active:
        flags.append("auto_exposure_active")
        reliability -= 0.15
    if str(camera.hdr_mode or "").lower() in {"on", "auto", "unknown"}:
        flags.append("hdr_or_night_processing_possible")
        reliability -= 0.08
    if camera.night_mode:
        flags.append("night_mode_active")
        reliability -= 0.08
    if (camera.digital_zoom or 1.0) > 1.01:
        flags.append("digital_zoom_active")
        reliability -= 0.1
    return reliability, flags


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


class CaptureNormalizer:
    """Module 1 deterministic head with the same output contract as future learned heads."""

    def __init__(self, config: CaptureNormalizationConfig | None = None) -> None:
        self.config = config or CaptureNormalizationConfig()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "CaptureNormalizer":
        return cls(CaptureNormalizationConfig.from_checkpoint(path))

    def normalize_array(self, rgb01: np.ndarray, frame: FrameRecord) -> NormalizedFrameProduct:
        if rgb01.ndim != 3 or rgb01.shape[2] != 3:
            raise ValueError("capture normalization expects an RGB image array with shape HxWx3")
        height, width = rgb01.shape[:2]
        camera = frame.camera
        factor = exposure_factor(camera, self.config)
        linear_luma = srgb_inverse_luma(rgb01)
        if self.config.response_gamma != 1.0:
            linear_luma = np.clip(linear_luma, 0.0, 1.0) ** max(0.05, self.config.response_gamma)
        luma_proxy = np.clip(linear_luma / factor, 0.0, 1.0).astype(np.float32)

        max_channel = np.max(rgb01, axis=2)
        saturation_mask = max_channel >= self.config.saturation_threshold
        glare_mask = max_channel >= self.config.glare_threshold
        bloom_seed = max_channel >= self.config.bloom_threshold
        bloom_mask = _dilate(bloom_seed, self.config.bloom_radius_px) & ~saturation_mask

        base_reliability, flags = _quality_penalty(camera)
        reliability_mask = np.full((height, width), base_reliability, dtype=np.float32)
        reliability_mask[saturation_mask] *= 0.15
        reliability_mask[bloom_mask] *= 0.45
        reliability_mask[glare_mask & ~saturation_mask] *= 0.65
        reliability_mask = np.clip(reliability_mask, self.config.min_reliability, 1.0)
        radiometric_uncertainty = (1.0 - reliability_mask).astype(np.float32)

        saturation_fraction = float(np.mean(saturation_mask))
        bloom_fraction = float(np.mean(bloom_mask))
        glare_fraction = float(np.mean(glare_mask))
        if saturation_fraction > 0.001:
            flags.append("saturation")
        if bloom_fraction > 0.002:
            flags.append("bloom")
        if glare_fraction > 0.005:
            flags.append("glare")
        if factor < 0.25 or factor > 8.0:
            flags.append("extreme_exposure_factor")

        return NormalizedFrameProduct(
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp_ns,
            width=width,
            height=height,
            luma_proxy=luma_proxy,
            reliability_mask=reliability_mask,
            radiometric_uncertainty=radiometric_uncertainty,
            saturation_mask=saturation_mask,
            bloom_mask=bloom_mask,
            glare_mask=glare_mask,
            exposure_factor=factor,
            reliability_score=float(np.mean(reliability_mask)),
            quality_flags=tuple(sorted(set(flags))),
            metadata={
                "implementation": self.config.implementation,
                "checkpoint_path": self.config.checkpoint_path,
                "metadata_quality": camera.metadata_quality,
                "ae_mode": camera.ae_mode,
                "hdr_mode": camera.hdr_mode,
                "night_mode": camera.night_mode,
            },
        )

    def normalize_path(self, image_path: str | Path, frame: FrameRecord) -> NormalizedFrameProduct:
        return self.normalize_array(load_rgb01(image_path), frame)
