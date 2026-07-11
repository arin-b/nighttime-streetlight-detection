from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rbccps_measurement.fusion.monotonic_heads import FusionResult


@dataclass(frozen=True)
class ModelSlotFusionEvidence:
    adjusted_result: FusionResult
    metrics: dict[str, float | str | list[str]]
    flags: list[str]


def _category(score: float) -> str:
    if score >= 0.72:
        return "adequate"
    if score >= 0.45:
        return "marginal"
    if score >= 0.2:
        return "poor"
    return "unknown"


def _float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fuse_model_slots(base: FusionResult, slot_metrics: dict[str, Any] | None, detector_score: float) -> ModelSlotFusionEvidence:
    if not slot_metrics:
        return ModelSlotFusionEvidence(
            adjusted_result=base,
            metrics={"model_slot_fusion_active": 0.0},
            flags=[],
        )

    lowlight = slot_metrics.get("lowlight_zero_dce_epoch99", {})
    retinex = slot_metrics.get("retinex_decom_9200", {})
    feature = slot_metrics.get("feature_resnet18_imagenet", {})
    segmentation = slot_metrics.get("segmentation_deeplabv3_mobilenet_v3", {})

    input_luma = _float(lowlight, "input_luma_mean")
    enhanced_luma = _float(lowlight, "enhanced_luma_mean")
    lowlight_gain = max(0.0, min(1.0, enhanced_luma - input_luma))
    illumination_mean = max(0.0, min(1.0, _float(retinex, "illumination_mean")))
    reflectance_mean = max(0.0, min(1.0, _float(retinex, "reflectance_mean")))
    embedding_norm = max(0.0, min(1.0, _float(feature, "embedding_l2_norm") / 25.0))

    histogram = segmentation.get("segmentation_class_histogram", {}) if isinstance(segmentation, dict) else {}
    if isinstance(histogram, dict):
        non_background = max(0.0, min(1.0, 1.0 - float(histogram.get("0", 1.0))))
    else:
        non_background = 0.0
    public_space_support = max(0.15, min(1.0, 0.5 + non_background))

    detector_support = max(0.0, min(1.0, detector_score))
    learned_illumination_support = (
        0.34 * detector_support
        + 0.24 * illumination_mean
        + 0.16 * lowlight_gain
        + 0.14 * reflectance_mean
        + 0.08 * embedding_norm
        + 0.04 * public_space_support
    )

    detector_penalty = max(0.0, 0.30 - detector_support) * 0.75
    lowlight_uncertainty = 0.06 if input_luma < 0.12 else 0.0
    slot_adjustment = 0.42 * (learned_illumination_support - 0.35) - detector_penalty - lowlight_uncertainty
    adjusted_score = max(0.0, min(1.0, base.overall_score + slot_adjustment))

    confidence_delta = 0.20 * detector_support + 0.08 * embedding_norm - 0.18 * (1.0 - public_space_support)
    if detector_support < 0.20:
        confidence_delta -= 0.22
    adjusted_confidence = max(0.0, min(1.0, base.confidence + confidence_delta))

    flags = ["model_slot_fusion_active"]
    if detector_support < 0.20:
        flags.append("low_detector_slot_confidence")
    if non_background < 0.01:
        flags.append("segmentation_slot_low_scene_structure")
    if input_luma < 0.12:
        flags.append("very_low_luma_slot_input")

    return ModelSlotFusionEvidence(
        adjusted_result=FusionResult(
            overall_score=adjusted_score,
            overall_category=_category(adjusted_score),
            confidence=adjusted_confidence,
            fusion_output=base.fusion_output,
        ),
        metrics={
            "model_slot_fusion_active": 1.0,
            "slot_detector_support": round(detector_support, 4),
            "slot_lowlight_gain": round(lowlight_gain, 4),
            "slot_retinex_illumination_mean": round(illumination_mean, 4),
            "slot_retinex_reflectance_mean": round(reflectance_mean, 4),
            "slot_feature_embedding_norm01": round(embedding_norm, 4),
            "slot_public_space_support": round(public_space_support, 4),
            "slot_learned_illumination_support": round(learned_illumination_support, 4),
            "slot_fusion_score_adjustment": round(slot_adjustment, 4),
        },
        flags=flags,
    )
