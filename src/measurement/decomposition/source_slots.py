from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rbccps_measurement.contracts.input_schema import DetectorTrackRecord, FrameRecord
from rbccps_measurement.contracts.module_io import AffectedRegionFieldOutput, NormalizedFrameProduct, SegmentationMaskOutput, SourceFieldOutput
from rbccps_measurement.normalization.module1 import CaptureNormalizer


SOURCE_CLASSES = (
    "target_lamp",
    "other_lamp",
    "headlight",
    "shopfront_or_window",
    "sign_or_signal",
    "reflection",
    "unknown_bright_source",
)


@dataclass(frozen=True)
class SourceDecompositionConfig:
    implementation: str = "deterministic_source_slots_v1"
    target_prior_weight: float = 1.25
    other_lamp_prior_weight: float = 0.35
    headlight_prior_weight: float = 0.85
    shopfront_prior_weight: float = 0.75
    signal_prior_weight: float = 0.70
    reflection_prior_weight: float = 0.80
    unknown_prior_weight: float = 0.28
    residual_tolerance: float = 0.18

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "target_prior_weight": self.target_prior_weight,
            "other_lamp_prior_weight": self.other_lamp_prior_weight,
            "headlight_prior_weight": self.headlight_prior_weight,
            "shopfront_prior_weight": self.shopfront_prior_weight,
            "signal_prior_weight": self.signal_prior_weight,
            "reflection_prior_weight": self.reflection_prior_weight,
            "unknown_prior_weight": self.unknown_prior_weight,
            "residual_tolerance": self.residual_tolerance,
        }


@dataclass(frozen=True)
class SourceEvidence:
    target_lamp: float
    other_lamps: float
    headlights: float
    shopfronts: float
    reflections: float
    unknown: float
    sign_or_signal: float = 0.0
    field_output: SourceFieldOutput | None = None

    @property
    def confounder_penalty(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                self.headlights
                + self.shopfronts
                + self.reflections
                + self.sign_or_signal
                + 0.5 * self.unknown,
            ),
        )


def _resolve_image_path(frame: FrameRecord, frame_root: str | Path) -> Path:
    path = Path(frame.image_uri)
    return path if path.is_absolute() else Path(frame_root) / path


def _load_rgb01(frame: FrameRecord, frame_root: str | Path) -> np.ndarray | None:
    path = _resolve_image_path(frame, frame_root)
    if not path.exists():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _clip01(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def _luma_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)


def _bbox_pixels(track: DetectorTrackRecord, frame: FrameRecord) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = track.bbox_xyxy
    if track.bbox_format == "normalized_xyxy_original_frame":
        return x1 * frame.width, y1 * frame.height, x2 * frame.width, y2 * frame.height
    return x1, y1, x2, y2


def _track_blob(track: DetectorTrackRecord, frame: FrameRecord) -> np.ndarray:
    x1, y1, x2, y2 = _bbox_pixels(track, frame)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    yy, xx = np.mgrid[0:frame.height, 0:frame.width].astype(np.float32)
    sigma_x = max(3.0, width * 0.75)
    sigma_y = max(3.0, height * 0.95)
    return np.exp(-(((xx - center_x) ** 2) / (2 * sigma_x * sigma_x) + ((yy - center_y) ** 2) / (2 * sigma_y * sigma_y))).astype(np.float32)


def _source_probabilities(fields: dict[str, np.ndarray]) -> dict[str, float]:
    masses = {name: float(np.sum(field)) for name, field in fields.items()}
    total = sum(masses.values())
    if total <= 1e-9:
        return {name: round(1.0 / len(fields), 6) for name in fields}
    return {name: round(value / total, 6) for name, value in masses.items()}


def _normalize_luma_budget(luma: np.ndarray, priors: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    stack = np.stack([np.maximum(0.0, priors[name]) for name in SOURCE_CLASSES], axis=0).astype(np.float32)
    prior_sum = np.sum(stack, axis=0)
    fields: dict[str, np.ndarray] = {}
    for index, name in enumerate(SOURCE_CLASSES):
        share = np.divide(stack[index], prior_sum, out=np.zeros_like(luma, dtype=np.float32), where=prior_sum > 1e-9)
        fields[name] = _clip01(luma * share)
    reconstruction = np.sum(np.stack([fields[name] for name in SOURCE_CLASSES], axis=0), axis=0)
    residual = np.maximum(0.0, luma - reconstruction).astype(np.float32)
    fields["unknown_bright_source"] = _clip01(fields["unknown_bright_source"] + residual)
    reconstruction = np.sum(np.stack([fields[name] for name in SOURCE_CLASSES], axis=0), axis=0)
    residual = np.abs(luma - reconstruction).astype(np.float32)
    reconstruction_error = float(np.mean(residual))
    return fields, residual, reconstruction_error


def deterministic_source_decomposition(
    track_id: str,
    tracks: list[DetectorTrackRecord],
    frames: dict[int, FrameRecord],
    frame_root: str | Path = ".",
    normalized_product: NormalizedFrameProduct | None = None,
    segmentation: SegmentationMaskOutput | None = None,
    affected_region: AffectedRegionFieldOutput | None = None,
    status_confidence: float = 0.65,
    config: SourceDecompositionConfig | None = None,
) -> SourceFieldOutput:
    config = config or SourceDecompositionConfig()
    records = sorted(tracks, key=lambda item: (item.timestamp_ns, item.frame_id))
    if not records:
        raise ValueError("source decomposition requires at least one track record")
    evidence_track = records[len(records) // 2]
    frame = frames[evidence_track.frame_id]
    flags: list[str] = ["deterministic_source_slots"]

    if normalized_product is None:
        rgb = _load_rgb01(frame, frame_root)
        if rgb is None:
            flags.append("image_missing")
            luma = np.zeros((frame.height, frame.width), dtype=np.float32)
            saturation = glare = bloom = np.zeros_like(luma)
        else:
            try:
                normalized_product = CaptureNormalizer().normalize_array(rgb, frame)
            except Exception:
                flags.append("normalization_product_unavailable")
                luma = _luma_from_rgb(rgb)
                saturation = glare = bloom = np.zeros_like(luma)
    if normalized_product is not None:
        luma = normalized_product.luma_proxy.astype(np.float32)
        saturation = normalized_product.saturation_mask.astype(np.float32)
        glare = normalized_product.glare_mask.astype(np.float32)
        bloom = normalized_product.bloom_mask.astype(np.float32)

    shape = luma.shape
    zero = np.zeros(shape, dtype=np.float32)
    semantic = segmentation.semantic_masks if segmentation is not None else {}
    affected = affected_region.affected_field.astype(np.float32) if affected_region is not None else np.ones(shape, dtype=np.float32)
    public_space = affected_region.public_space_mask.astype(np.float32) if affected_region is not None else np.ones(shape, dtype=np.float32)
    target_blob = _track_blob(evidence_track, frame)

    target_prior = config.target_prior_weight * target_blob * affected * public_space * max(0.05, status_confidence)
    other_blob = zero.copy()
    for track in records:
        if track.track_id != evidence_track.track_id:
            other_blob = np.maximum(other_blob, _track_blob(track, frame) * float(track.detector_score))
    other_prior = config.other_lamp_prior_weight * other_blob

    vehicle = semantic.get("vehicle", zero)
    shopfront = np.maximum(semantic.get("shopfront", zero), semantic.get("window", zero))
    signal = np.maximum(semantic.get("sign_billboard", zero), semantic.get("traffic_signal", zero))
    reflection = np.maximum(semantic.get("wet_reflection_like_road", zero), np.maximum(saturation, glare) * public_space)
    bright_artifact = np.maximum.reduce([saturation, glare, bloom])
    confounder_candidate = segmentation.confounder_candidate_mask.astype(np.float32) if segmentation is not None else bright_artifact

    priors = {
        "target_lamp": _clip01(target_prior),
        "other_lamp": _clip01(other_prior),
        "headlight": _clip01(config.headlight_prior_weight * vehicle * luma),
        "shopfront_or_window": _clip01(config.shopfront_prior_weight * shopfront * (0.25 + luma)),
        "sign_or_signal": _clip01(config.signal_prior_weight * signal * (0.35 + luma)),
        "reflection": _clip01(config.reflection_prior_weight * reflection * (0.35 + luma)),
        "unknown_bright_source": _clip01(config.unknown_prior_weight * np.maximum(confounder_candidate, bright_artifact) * (0.3 + luma)),
    }
    prior_total = np.sum(np.stack([priors[name] for name in SOURCE_CLASSES], axis=0), axis=0)
    unexplained = np.maximum(0.0, luma - prior_total)
    priors["unknown_bright_source"] = _clip01(priors["unknown_bright_source"] + unexplained * (luma > 0.55))

    fields, residual, reconstruction_error = _normalize_luma_budget(luma, priors)
    probabilities = _source_probabilities(fields)
    non_target = sum(probabilities[name] for name in SOURCE_CLASSES if name != "target_lamp")
    confounder_penalty = max(
        0.0,
        min(
            1.0,
            probabilities["headlight"]
            + probabilities["shopfront_or_window"]
            + probabilities["sign_or_signal"]
            + probabilities["reflection"]
            + 0.5 * probabilities["unknown_bright_source"],
        ),
    )
    source_confusion_score = max(0.0, min(1.0, non_target * (1.0 - probabilities["target_lamp"]) + reconstruction_error))
    if reconstruction_error > config.residual_tolerance:
        flags.append("high_source_reconstruction_residual")
    if confounder_penalty > 0.45:
        flags.append("mixed_light_sources")

    return SourceFieldOutput(
        frame_id=frame.frame_id,
        track_id=track_id,
        source_fields=fields,
        source_probabilities=probabilities,
        residual_field=residual,
        reconstruction_error=reconstruction_error,
        confounder_penalty=confounder_penalty,
        source_confusion_score=source_confusion_score,
        quality_flags=tuple(sorted(set(flags))),
        metadata={
            "implementation": config.implementation,
            "source_classes": list(SOURCE_CLASSES),
            "additive_interpretation": "proxy_luma_budget_not_physical_light_transport",
        },
    )


def source_output_to_evidence(output: SourceFieldOutput) -> SourceEvidence:
    p = output.source_probabilities
    return SourceEvidence(
        target_lamp=float(p.get("target_lamp", 0.0)),
        other_lamps=float(p.get("other_lamp", 0.0)),
        headlights=float(p.get("headlight", 0.0)),
        shopfronts=float(p.get("shopfront_or_window", 0.0)),
        reflections=float(p.get("reflection", 0.0)),
        unknown=float(p.get("unknown_bright_source", 0.0)),
        sign_or_signal=float(p.get("sign_or_signal", 0.0)),
        field_output=output,
    )


def estimate_source_evidence(
    tracks: list[DetectorTrackRecord],
    frames: dict[int, FrameRecord],
    frame_root: str | Path = ".",
    normalized_product: NormalizedFrameProduct | None = None,
    segmentation: SegmentationMaskOutput | None = None,
    affected_region: AffectedRegionFieldOutput | None = None,
    status_confidence: float = 0.65,
) -> SourceEvidence:
    output = deterministic_source_decomposition(
        tracks[0].track_id,
        tracks,
        frames,
        frame_root=frame_root,
        normalized_product=normalized_product,
        segmentation=segmentation,
        affected_region=affected_region,
        status_confidence=status_confidence,
    )
    return source_output_to_evidence(output)


class SourceDecompositionEstimator:
    def __init__(self, checkpoint: dict[str, Any] | None = None, torch_model: Any | None = None) -> None:
        self.checkpoint = checkpoint or {"config": SourceDecompositionConfig().to_dict()}
        self.config = _config_from_checkpoint(self.checkpoint)
        self.torch_model = torch_model

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "SourceDecompositionEstimator":
        checkpoint_path = Path(path)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        weights_ref = checkpoint.get("weights")
        if weights_ref:
            weights_path = Path(weights_ref)
            if not weights_path.is_absolute():
                weights_path = checkpoint_path.parent / weights_path
            if weights_path.exists():
                try:
                    from rbccps_measurement.decomposition.torch_models import load_torch_source_model

                    return cls(checkpoint, torch_model=load_torch_source_model(weights_path, checkpoint))
                except Exception as exc:
                    checkpoint["torch_load_error"] = str(exc)
        return cls(checkpoint)

    def predict(self, *args: Any, **kwargs: Any) -> SourceFieldOutput:
        return deterministic_source_decomposition(*args, config=self.config, **kwargs)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> SourceDecompositionConfig:
    payload = checkpoint.get("config", {})
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
