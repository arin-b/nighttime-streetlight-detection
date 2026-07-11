from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class NormalizedFrameProduct:
    """Module 1 output consumed by every downstream measurement block."""

    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    luma_proxy: ArrayLike
    reliability_mask: ArrayLike
    radiometric_uncertainty: ArrayLike
    saturation_mask: ArrayLike
    bloom_mask: ArrayLike
    glare_mask: ArrayLike
    exposure_factor: float
    reliability_score: float
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "width": self.width,
            "height": self.height,
            "luma_mean": round(float(np.mean(self.luma_proxy)), 6),
            "luma_p50": round(float(np.quantile(self.luma_proxy, 0.50)), 6),
            "luma_p95": round(float(np.quantile(self.luma_proxy, 0.95)), 6),
            "reliability_score": round(float(self.reliability_score), 6),
            "saturation_fraction": round(float(np.mean(self.saturation_mask)), 6),
            "bloom_fraction": round(float(np.mean(self.bloom_mask)), 6),
            "glare_fraction": round(float(np.mean(self.glare_mask)), 6),
            "exposure_factor": round(float(self.exposure_factor), 6),
            "quality_flags": list(self.quality_flags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LampStatusOutput:
    track_id: str
    status_label: str
    probabilities: dict[str, float]
    dim_probability: float
    occluded_probability: float
    flicker_index: float
    saturated_flag: bool
    confidence: float
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LampCropSequence:
    """Module 2 crop-window artifact for one lamp track."""

    track_id: str
    crop_tensor: ArrayLike
    valid_mask: ArrayLike
    frame_ids: tuple[int, ...]
    timestamps_ns: tuple[int, ...]
    bbox_xyxy: ArrayLike
    metadata_tokens: ArrayLike
    token_names: tuple[str, ...]
    quality_flags: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        valid = self.valid_mask.astype(bool)
        crops = self.crop_tensor[valid] if np.any(valid) else self.crop_tensor[:0]
        return {
            "track_id": self.track_id,
            "sequence_length": int(self.crop_tensor.shape[0]),
            "valid_frames": int(np.sum(valid)),
            "crop_shape": list(self.crop_tensor.shape[1:]),
            "crop_luma_mean": round(float(np.mean(crops)), 6) if crops.size else 0.0,
            "quality_flags": list(self.quality_flags),
        }


@dataclass(frozen=True)
class LatentEmissionStateOutput:
    track_id: str
    status_label: str
    status_posterior: dict[str, float]
    emission_posterior: dict[str, float]
    occlusion_posterior: dict[str, float]
    capture_posterior: dict[str, float]
    flicker_posterior: dict[str, float]
    dim_probability: float
    occluded_probability: float
    flicker_index: float
    saturated_flag: bool
    confidence: float
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_lamp_status_output(self) -> LampStatusOutput:
        return LampStatusOutput(
            track_id=self.track_id,
            status_label=self.status_label,
            probabilities=self.status_posterior,
            dim_probability=self.dim_probability,
            occluded_probability=self.occluded_probability,
            flicker_index=self.flicker_index,
            saturated_flag=self.saturated_flag,
            confidence=self.confidence,
            quality_flags=self.quality_flags,
        )


@dataclass(frozen=True)
class SegmentationMaskOutput:
    frame_id: int
    semantic_masks: dict[str, ArrayLike]
    class_order: tuple[str, ...]
    public_space_mask: ArrayLike
    occluder_mask: ArrayLike
    confounder_mask: ArrayLike
    confounder_candidate_mask: ArrayLike
    uncertainty_map: ArrayLike
    enhanced_view_used: bool = False
    confidence: float = 0.0
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AffectedRegionFieldOutput:
    track_id: str
    frame_id: int
    affected_field: ArrayLike
    public_space_mask: ArrayLike
    road_region: ArrayLike
    footpath_region: ArrayLike
    crossing_region: ArrayLike
    verge_region: ArrayLike
    occlusion_gate: ArrayLike
    uncertainty_map: ArrayLike
    region_mix: dict[str, float]
    mask_ref: str
    quality: str
    geometry_quality: float
    field_confidence: float = 0.0
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceFieldOutput:
    frame_id: int
    track_id: str
    source_fields: dict[str, ArrayLike]
    source_probabilities: dict[str, float]
    residual_field: ArrayLike
    reconstruction_error: float
    confounder_penalty: float
    source_confusion_score: float
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RISDecompositionOutput:
    frame_id: int
    reflectance_like: ArrayLike
    illumination_like: ArrayLike
    source_like: ArrayLike
    reconstruction_proxy: ArrayLike
    confidence_map: ArrayLike
    reconstruction_error: float
    decomposition_confidence: float
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UsefulFeatureVector:
    track_id: str
    coverage_proxy: float
    adequacy_proxy: float
    uniformity_proxy: float
    dark_hole_fraction: float
    glare_penalty: float
    confounder_penalty: float
    occlusion_penalty: float
    temporal_stability: float
    quantiles: dict[str, float] = field(default_factory=dict)
    dark_hole_probability_map: ArrayLike | None = None
    utility_score: float = 0.0
    quality_by_region: dict[str, float] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttributionOutput:
    track_id: str
    attribution_score: float
    attribution_class: str
    counterfactual_uncertainty: float
    all_source_utility: float = 0.0
    without_target_utility: float = 0.0
    source_competition: dict[str, float] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FusionOutput:
    track_id: str
    useful_illumination_score: float
    useful_illumination_class: str
    confidence: float
    prediction_set: tuple[str, ...]
    abstention_action: str
    uncertainty_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneGraphArtifact:
    track_id: str
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    graph_features: dict[str, float]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneGraphFusionOutput:
    track_id: str
    graph: SceneGraphArtifact
    component_scores: dict[str, float]
    monotonic_contributions: dict[str, float]
    raw_score: float
    ordinal_class: str
    raw_confidence: float
    uncertainty_index: float
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationAbstentionOutput:
    calibration_group_key: str
    calibrated_confidence: float
    prediction_set: tuple[str, ...]
    risk_estimate: float
    abstention_action: str
    physical_validity_score: float
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SparseLuxReference:
    clip_id: str
    frame_id: str
    track_id: str
    point_type: str
    lux_value: float | None
    x: float | None = None
    y: float | None = None
    ground_x_m: float | None = None
    ground_y_m: float | None = None
    orientation: str = "horizontal"
    source_file: str = ""
    source_item: str = ""
    validation_status: str = "valid"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhotometricFieldOutput:
    track_id: str
    clip_id: str
    q_signal: float
    q_calibration: float
    q_calib: float
    tau_required: float
    physical_valid: bool
    reason: str
    horizontal_illuminance_lux_mean: float | None
    horizontal_illuminance_lux_interval: tuple[float, float] | None
    vertical_illuminance_lux_mean: float | None
    vertical_illuminance_lux_interval: tuple[float, float] | None
    served_area_m2_est: float | None
    reference_summary: dict[str, Any]
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteGraphArtifact:
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    graph_statistics: dict[str, float]
    route_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregatedLampReport:
    candidate_lamp_id: str
    contributing_observations: tuple[str, ...]
    contributing_clips: tuple[str, ...]
    geo_summary: dict[str, Any]
    consensus_metrics: dict[str, Any]
    category_histogram: dict[str, float]
    physical_estimate_summary: dict[str, Any]
    disagreement_score: float
    manual_review_priority: str
    quality_flags: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoadSegmentReport:
    segment_id: str
    route_group: str
    observation_count: int
    candidate_lamp_count: int
    mean_score: float
    worst_category: str
    underlighting_score: float
    manual_review_priority: str
    quality_flags: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditTrailArtifact:
    source_reports: tuple[dict[str, Any], ...]
    module_versions: dict[str, Any]
    evidence_refs: tuple[dict[str, Any], ...]
    quality_flags: tuple[str, ...]
    calibration_summary: dict[str, Any]
    generation_metadata: dict[str, Any] = field(default_factory=dict)
