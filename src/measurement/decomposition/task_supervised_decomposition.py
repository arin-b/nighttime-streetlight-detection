from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rbccps_measurement.contracts.input_schema import FrameRecord
from rbccps_measurement.contracts.module_io import NormalizedFrameProduct, RISDecompositionOutput, SegmentationMaskOutput, SourceFieldOutput
from rbccps_measurement.normalization.module1 import CaptureNormalizer


RIS_INTERPRETATION = "learned_measurement_representation_not_physical_truth"


@dataclass(frozen=True)
class RISDecompositionConfig:
    implementation: str = "deterministic_ris_decomposition_v1"
    blur_radius: int = 5
    source_threshold: float = 0.18
    min_confidence: float = 0.05
    interpretation: str = RIS_INTERPRETATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "blur_radius": self.blur_radius,
            "source_threshold": self.source_threshold,
            "min_confidence": self.min_confidence,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class DecompositionSpec:
    pretrained_asset: str = "retinex_decom_9200"
    interpretation: str = RIS_INTERPRETATION


def _resolve_image_path(frame: FrameRecord, frame_root: str | Path) -> Path:
    path = Path(frame.image_uri)
    return path if path.is_absolute() else Path(frame_root) / path


def _load_rgb01(frame: FrameRecord, frame_root: str | Path) -> np.ndarray | None:
    path = _resolve_image_path(frame, frame_root)
    if not path.exists():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _luma_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)


def _clip01(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return array.astype(np.float32)
    result = np.zeros_like(array, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(array, dy, axis=0), dx, axis=1)
            count += 1
    return (result / max(1, count)).astype(np.float32)


def deterministic_ris_decomposition(
    frame: FrameRecord,
    frame_root: str | Path = ".",
    normalized_product: NormalizedFrameProduct | None = None,
    segmentation: SegmentationMaskOutput | None = None,
    source_output: SourceFieldOutput | None = None,
    config: RISDecompositionConfig | None = None,
) -> RISDecompositionOutput:
    config = config or RISDecompositionConfig()
    flags: list[str] = ["deterministic_ris_decomposition", "non_physical_proxy_representation"]
    rgb = _load_rgb01(frame, frame_root)
    if rgb is None:
        flags.append("image_missing")
        rgb = np.zeros((frame.height, frame.width, 3), dtype=np.float32)

    if normalized_product is None:
        try:
            normalized_product = CaptureNormalizer().normalize_array(rgb, frame)
        except Exception:
            flags.append("normalization_product_unavailable")
            luma = _luma_from_rgb(rgb)
            reliability = np.full_like(luma, 0.45, dtype=np.float32)
    if normalized_product is not None:
        luma = normalized_product.luma_proxy.astype(np.float32)
        reliability = normalized_product.reliability_mask.astype(np.float32)

    illumination_like = _clip01(_box_blur(luma, config.blur_radius))
    local_floor = np.maximum(illumination_like, 1e-3)
    reflectance_gray = _clip01(luma / local_floor)
    if segmentation is not None:
        structure_support = 1.0 - np.clip(segmentation.uncertainty_map.astype(np.float32), 0.0, 1.0)
        reflectance_gray = _clip01(0.72 * reflectance_gray + 0.28 * structure_support)

    reflectance_like = np.repeat(reflectance_gray[:, :, None], 3, axis=2)
    if source_output is not None:
        source_like = _clip01(
            np.maximum.reduce(
                [
                    source_output.source_fields.get("target_lamp", np.zeros_like(luma)),
                    source_output.source_fields.get("other_lamp", np.zeros_like(luma)),
                    source_output.source_fields.get("headlight", np.zeros_like(luma)),
                    source_output.source_fields.get("shopfront_or_window", np.zeros_like(luma)),
                    source_output.source_fields.get("sign_or_signal", np.zeros_like(luma)),
                    source_output.source_fields.get("reflection", np.zeros_like(luma)),
                    source_output.source_fields.get("unknown_bright_source", np.zeros_like(luma)),
                ]
            )
        )
    else:
        source_like = _clip01(np.maximum(0.0, luma - illumination_like) * (luma > config.source_threshold))
        flags.append("source_output_missing")

    reconstruction_luma = _clip01(illumination_like * np.mean(reflectance_like, axis=2) + 0.35 * source_like)
    reconstruction_proxy = _clip01(rgb * reconstruction_luma[:, :, None] / np.maximum(luma[:, :, None], 1e-3))
    reconstruction_error = float(np.mean(np.abs(luma - reconstruction_luma)))
    confidence_map = _clip01(reliability * (1.0 - np.minimum(1.0, np.abs(luma - reconstruction_luma) * 2.0)))
    if source_output is not None:
        confidence_map = _clip01(confidence_map * (1.0 - 0.35 * source_output.source_confusion_score))
    decomposition_confidence = float(np.clip(np.mean(confidence_map), config.min_confidence, 1.0))
    if reconstruction_error > 0.25:
        flags.append("high_ris_reconstruction_error")
    if decomposition_confidence < 0.35:
        flags.append("low_ris_confidence")

    return RISDecompositionOutput(
        frame_id=frame.frame_id,
        reflectance_like=reflectance_like.astype(np.float32),
        illumination_like=illumination_like.astype(np.float32),
        source_like=source_like.astype(np.float32),
        reconstruction_proxy=reconstruction_proxy.astype(np.float32),
        confidence_map=confidence_map.astype(np.float32),
        reconstruction_error=reconstruction_error,
        decomposition_confidence=decomposition_confidence,
        quality_flags=tuple(sorted(set(flags))),
        metadata={
            "implementation": config.implementation,
            "interpretation": config.interpretation,
            "physical_claim": False,
            "supervision_roles": {
                "reflectance_like": "semantic_structure_support",
                "illumination_like": "coverage_and_adequacy_support",
                "source_like": "glare_and_confounder_support",
            },
        },
    )


class RISDecompositionEstimator:
    def __init__(self, checkpoint: dict[str, Any] | None = None, torch_model: Any | None = None) -> None:
        self.checkpoint = checkpoint or {"config": RISDecompositionConfig().to_dict()}
        self.config = _config_from_checkpoint(self.checkpoint)
        self.torch_model = torch_model

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "RISDecompositionEstimator":
        checkpoint_path = Path(path)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        weights_ref = checkpoint.get("weights")
        if weights_ref:
            weights_path = Path(weights_ref)
            if not weights_path.is_absolute():
                weights_path = checkpoint_path.parent / weights_path
            if weights_path.exists():
                try:
                    from rbccps_measurement.decomposition.torch_models import load_torch_ris_model

                    return cls(checkpoint, torch_model=load_torch_ris_model(weights_path, checkpoint))
                except Exception as exc:
                    checkpoint["torch_load_error"] = str(exc)
        return cls(checkpoint)

    def predict(self, *args: Any, **kwargs: Any) -> RISDecompositionOutput:
        return deterministic_ris_decomposition(*args, config=self.config, **kwargs)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> RISDecompositionConfig:
    payload = checkpoint.get("config", {})
    return RISDecompositionConfig(
        implementation=str(payload.get("implementation", "deterministic_ris_decomposition_v1")),
        blur_radius=int(payload.get("blur_radius", 5)),
        source_threshold=float(payload.get("source_threshold", 0.18)),
        min_confidence=float(payload.get("min_confidence", 0.05)),
        interpretation=str(payload.get("interpretation", RIS_INTERPRETATION)),
    )
