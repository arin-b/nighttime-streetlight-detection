from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rbccps_measurement.contracts.input_schema import FrameRecord
from rbccps_measurement.contracts.module_io import NormalizedFrameProduct, SegmentationMaskOutput
from rbccps_measurement.normalization.module1 import CaptureNormalizer


SEMANTIC_CLASSES = (
    "road",
    "footpath",
    "crossing",
    "curb",
    "median",
    "verge",
    "vegetation",
    "vehicle",
    "building_frontage",
    "shopfront",
    "window",
    "sign_billboard",
    "traffic_signal",
    "sky",
    "wet_reflection_like_road",
    "occluder",
    "unknown",
)
PUBLIC_SPACE_CLASSES = ("road", "footpath", "crossing", "curb", "median", "verge", "wet_reflection_like_road")
CONFOUNDER_CLASSES = ("vehicle", "shopfront", "window", "sign_billboard", "traffic_signal", "wet_reflection_like_road")
OCCLUDER_CLASSES = ("vegetation", "vehicle", "building_frontage", "sign_billboard", "occluder")


@dataclass(frozen=True)
class SegmentationConfig:
    implementation: str = "deterministic_illumination_disentangled_v1"
    uncertainty_floor: float = 0.08
    saturation_uncertainty_weight: float = 0.45
    glare_confounder_weight: float = 0.65
    enhanced_view_policy: str = "auxiliary_only"
    measurement_source: str = "original_or_normalized_capture"

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "uncertainty_floor": self.uncertainty_floor,
            "saturation_uncertainty_weight": self.saturation_uncertainty_weight,
            "glare_confounder_weight": self.glare_confounder_weight,
            "enhanced_view_policy": self.enhanced_view_policy,
            "measurement_source": self.measurement_source,
        }


def _resolve_image_path(frame: FrameRecord, frame_root: str | Path) -> Path:
    path = Path(frame.image_uri)
    return path if path.is_absolute() else Path(frame_root) / path


def _load_rgb01(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return yy / max(1, height - 1), xx / max(1, width - 1)


def _clip01(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def _sigmoid(value: np.ndarray, scale: float = 10.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scale * value))


def deterministic_segment_frame(
    frame: FrameRecord,
    frame_root: str | Path = ".",
    normalized_product: NormalizedFrameProduct | None = None,
    enhanced_image_path: str | Path | None = None,
    config: SegmentationConfig | None = None,
) -> SegmentationMaskOutput:
    """Produce structural masks without using enhancement as measurement evidence.

    This fallback is intentionally conservative. It uses image geometry as the
    primary structural prior and uses brightness-derived Module-1 masks only for
    uncertainty and confounder candidates, not to erase dark public space.
    """

    config = config or SegmentationConfig()
    image_path = _resolve_image_path(frame, frame_root)
    rgb = _load_rgb01(image_path)
    flags: list[str] = ["deterministic_segmentation_prior"]
    if rgb is None:
        flags.append("image_missing")
    if enhanced_image_path is not None:
        flags.append("enhanced_view_auxiliary_only")

    height, width = frame.height, frame.width
    y, x = _grid(height, width)
    center = 1.0 - np.abs(x - 0.5) * 2.0
    lower = _sigmoid(y - 0.52, scale=12.0)
    upper = 1.0 - lower

    road = _clip01(lower * (0.25 + 0.75 * center))
    footpath = _clip01(lower * (1.0 - center) * 0.85)
    curb = _clip01(np.exp(-((y - 0.58) ** 2) / 0.0018) * (0.25 + 0.75 * (1.0 - center)))
    crossing = _clip01(np.exp(-((y - 0.74) ** 2) / 0.0035) * center * 0.28)
    median = _clip01(lower * np.exp(-((x - 0.5) ** 2) / 0.0018) * 0.22)
    verge = _clip01(lower * ((x < 0.12) | (x > 0.88)).astype(np.float32) * 0.55)
    sky = _clip01(upper * _sigmoid(0.32 - y, scale=16.0))
    building = _clip01(upper * (1.0 - center) * 0.75)
    vegetation = _clip01(((x < 0.18) | (x > 0.82)).astype(np.float32) * np.exp(-((y - 0.42) ** 2) / 0.055) * 0.35)
    vehicle = _clip01(lower * center * np.exp(-((y - 0.70) ** 2) / 0.015) * 0.18)

    if normalized_product is not None:
        saturation = normalized_product.saturation_mask.astype(np.float32)
        bloom = normalized_product.bloom_mask.astype(np.float32)
        glare = normalized_product.glare_mask.astype(np.float32)
        reliability = normalized_product.reliability_mask.astype(np.float32)
    else:
        saturation = np.zeros((height, width), dtype=np.float32)
        bloom = np.zeros((height, width), dtype=np.float32)
        glare = np.zeros((height, width), dtype=np.float32)
        reliability = np.full((height, width), 0.45 if rgb is None else 0.62, dtype=np.float32)
        if rgb is not None:
            try:
                product = CaptureNormalizer().normalize_array(rgb, frame)
                saturation = product.saturation_mask.astype(np.float32)
                bloom = product.bloom_mask.astype(np.float32)
                glare = product.glare_mask.astype(np.float32)
                reliability = product.reliability_mask.astype(np.float32)
            except Exception:
                flags.append("normalization_product_unavailable")

    bright_artifact = _clip01(np.maximum.reduce([saturation, bloom, glare]))
    wet_reflection = _clip01(road * bright_artifact * 0.8)
    shopfront = _clip01(building * np.maximum(0.0, x - 0.58) * 0.28)
    window = _clip01(building * 0.22)
    sign_billboard = _clip01(building * np.exp(-((y - 0.36) ** 2) / 0.015) * 0.16 + bright_artifact * 0.12)
    traffic_signal = _clip01(np.exp(-((x - 0.72) ** 2 + (y - 0.42) ** 2) / 0.0025) * 0.20)
    occluder = _clip01(np.maximum.reduce([vegetation * 0.9, vehicle * 0.85, sign_billboard * 0.45]))
    unknown = _clip01(1.0 - np.maximum.reduce([road, footpath, crossing, curb, median, verge, sky, building]))

    semantic_masks = {
        "road": road,
        "footpath": footpath,
        "crossing": crossing,
        "curb": curb,
        "median": median,
        "verge": verge,
        "vegetation": vegetation,
        "vehicle": vehicle,
        "building_frontage": building,
        "shopfront": shopfront,
        "window": window,
        "sign_billboard": sign_billboard,
        "traffic_signal": traffic_signal,
        "sky": sky,
        "wet_reflection_like_road": wet_reflection,
        "occluder": occluder,
        "unknown": unknown,
    }
    public_space_mask = _clip01(np.maximum.reduce([semantic_masks[label] for label in PUBLIC_SPACE_CLASSES]))
    confounder_candidate_mask = _clip01(
        np.maximum.reduce([semantic_masks[label] for label in CONFOUNDER_CLASSES] + [bright_artifact * config.glare_confounder_weight])
    )
    occluder_mask = _clip01(np.maximum.reduce([semantic_masks[label] for label in OCCLUDER_CLASSES]))
    uncertainty = _clip01(
        config.uncertainty_floor
        + 0.35 * semantic_masks["unknown"]
        + config.saturation_uncertainty_weight * bright_artifact
        + 0.35 * (1.0 - reliability)
    )
    if rgb is None:
        uncertainty = _clip01(np.maximum(uncertainty, 0.55))
    confidence = float(np.clip(1.0 - np.mean(uncertainty), 0.0, 1.0))

    return SegmentationMaskOutput(
        frame_id=frame.frame_id,
        semantic_masks=semantic_masks,
        class_order=SEMANTIC_CLASSES,
        public_space_mask=public_space_mask,
        occluder_mask=occluder_mask,
        confounder_mask=confounder_candidate_mask,
        confounder_candidate_mask=confounder_candidate_mask,
        uncertainty_map=uncertainty,
        enhanced_view_used=enhanced_image_path is not None,
        confidence=confidence,
        quality_flags=tuple(sorted(set(flags))),
        metadata={
            "implementation": config.implementation,
            "enhanced_view_policy": config.enhanced_view_policy,
            "measurement_source": config.measurement_source,
        },
    )


class SegmentationEstimator:
    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoint = checkpoint or {"config": SegmentationConfig().to_dict()}
        self.config = _config_from_checkpoint(self.checkpoint)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "SegmentationEstimator":
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(checkpoint)

    def predict(
        self,
        frame: FrameRecord,
        frame_root: str | Path = ".",
        normalized_product: NormalizedFrameProduct | None = None,
        enhanced_image_path: str | Path | None = None,
    ) -> SegmentationMaskOutput:
        return deterministic_segment_frame(frame, frame_root, normalized_product, enhanced_image_path, self.config)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> SegmentationConfig:
    payload = checkpoint.get("config", {})
    return SegmentationConfig(
        implementation=str(payload.get("implementation", "deterministic_illumination_disentangled_v1")),
        uncertainty_floor=float(payload.get("uncertainty_floor", 0.08)),
        saturation_uncertainty_weight=float(payload.get("saturation_uncertainty_weight", 0.45)),
        glare_confounder_weight=float(payload.get("glare_confounder_weight", 0.65)),
        enhanced_view_policy=str(payload.get("enhanced_view_policy", "auxiliary_only")),
        measurement_source=str(payload.get("measurement_source", "original_or_normalized_capture")),
    )
