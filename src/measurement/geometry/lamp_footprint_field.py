from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbccps_measurement.contracts.input_schema import FrameRecord, DetectorTrackRecord
from rbccps_measurement.contracts.module_io import AffectedRegionFieldOutput, SegmentationMaskOutput
from rbccps_measurement.segmentation.illumination_disentangled import deterministic_segment_frame


@dataclass(frozen=True)
class FootprintEstimate:
    quality: str
    mask_ref: str
    geometry_quality: float
    field: AffectedRegionFieldOutput | None = None

    def to_dict(self, region_mix: dict[str, float]) -> dict[str, object]:
        mix = self.field.region_mix if self.field is not None else region_mix
        return {
            "quality": self.quality,
            "image_mask_uri": self.mask_ref,
            "ground_polygon_geojson": None,
            "region_mix": mix,
            "geometry_quality": round(self.geometry_quality, 4),
        }


@dataclass(frozen=True)
class FootprintConfig:
    implementation: str = "deterministic_lamp_conditioned_field_v1"
    distance_decay_fraction: float = 0.38
    downward_gate_strength: float = 12.0
    occlusion_suppression: float = 0.78
    weak_geometry_floor: float = 0.48

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "distance_decay_fraction": self.distance_decay_fraction,
            "downward_gate_strength": self.downward_gate_strength,
            "occlusion_suppression": self.occlusion_suppression,
            "weak_geometry_floor": self.weak_geometry_floor,
        }


def _geometry_quality(tracks: list[DetectorTrackRecord], frames: dict[int, FrameRecord]) -> float:
    pose_good = 0
    for track in tracks:
        pose = frames[track.frame_id].pose
        if pose.latitude is not None and pose.longitude is not None and (pose.gps_accuracy_m or 999) <= 10:
            pose_good += 1
    return pose_good / max(1, len(tracks))


def _quality_label(geometry_quality: float) -> str:
    if geometry_quality >= 0.75:
        return "good"
    if geometry_quality >= 0.35:
        return "moderate"
    return "weak"


def _bbox_pixels(track: DetectorTrackRecord, frame: FrameRecord) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = track.bbox_xyxy
    if track.bbox_format == "normalized_xyxy_original_frame":
        return x1 * frame.width, y1 * frame.height, x2 * frame.width, y2 * frame.height
    return x1, y1, x2, y2


def _normalize_mix(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 1e-9:
        return {key: 0.0 for key in values}
    return {key: round(max(0.0, value) / total, 4) for key, value in values.items()}


def estimate_affected_region_field(
    track_id: str,
    tracks: list[DetectorTrackRecord],
    frames: dict[int, FrameRecord],
    segmentation_by_frame: dict[int, SegmentationMaskOutput] | None = None,
    config: FootprintConfig | None = None,
) -> AffectedRegionFieldOutput:
    config = config or FootprintConfig()
    segmentation_by_frame = segmentation_by_frame or {}
    records = sorted(tracks, key=lambda item: (item.timestamp_ns, item.frame_id))
    if not records:
        raise ValueError("affected-region field requires at least one track record")
    evidence_track = records[len(records) // 2]
    frame = frames[evidence_track.frame_id]
    segmentation = segmentation_by_frame.get(frame.frame_id) or deterministic_segment_frame(frame)
    x1, y1, x2, y2 = _bbox_pixels(evidence_track, frame)
    lamp_x = (x1 + x2) / 2.0
    lamp_y = y2

    yy, xx = np.mgrid[0:frame.height, 0:frame.width].astype(np.float32)
    dx = (xx - lamp_x) / max(1.0, frame.width)
    dy = (yy - lamp_y) / max(1.0, frame.height)
    distance = np.sqrt(dx * dx + dy * dy)
    decay = np.exp(-distance / max(1e-3, config.distance_decay_fraction)).astype(np.float32)
    downward_gate = (1.0 / (1.0 + np.exp(-config.downward_gate_strength * dy))).astype(np.float32)

    public_space = segmentation.public_space_mask.astype(np.float32)
    occlusion_gate = np.clip(1.0 - config.occlusion_suppression * segmentation.occluder_mask.astype(np.float32), 0.0, 1.0).astype(np.float32)
    geometry_quality = _geometry_quality(records, frames)
    geometry_gate = config.weak_geometry_floor + (1.0 - config.weak_geometry_floor) * geometry_quality
    affected = np.clip(public_space * decay * downward_gate * occlusion_gate * geometry_gate, 0.0, 1.0).astype(np.float32)

    semantic = segmentation.semantic_masks
    road_region = affected * np.maximum(semantic.get("road", 0.0), semantic.get("wet_reflection_like_road", 0.0))
    footpath_region = affected * semantic.get("footpath", 0.0)
    crossing_region = affected * semantic.get("crossing", 0.0)
    verge_region = affected * np.maximum(semantic.get("verge", 0.0), semantic.get("curb", 0.0))
    region_mix = _normalize_mix(
        {
            "road": float(np.sum(road_region)),
            "footpath": float(np.sum(footpath_region)),
            "crossing": float(np.sum(crossing_region)),
            "verge": float(np.sum(verge_region)),
        }
    )
    uncertainty = np.clip(
        segmentation.uncertainty_map.astype(np.float32) * 0.55
        + (1.0 - occlusion_gate) * 0.25
        + (1.0 - geometry_quality) * 0.35,
        0.0,
        1.0,
    ).astype(np.float32)
    flags = list(segmentation.quality_flags)
    flags.append("deterministic_lamp_conditioned_field")
    if geometry_quality < 0.35:
        flags.append("weak_geometry")
    if float(np.max(public_space)) <= 0.0:
        flags.append("no_public_space_support")
    if float(np.max(affected)) <= 0.0:
        flags.append("empty_affected_field")
    field_confidence = float(np.clip(0.55 * geometry_quality + 0.30 * segmentation.confidence + 0.15 * (1.0 - np.mean(uncertainty)), 0.0, 1.0))

    return AffectedRegionFieldOutput(
        track_id=track_id,
        frame_id=frame.frame_id,
        affected_field=affected,
        public_space_mask=public_space,
        road_region=road_region.astype(np.float32),
        footpath_region=footpath_region.astype(np.float32),
        crossing_region=crossing_region.astype(np.float32),
        verge_region=verge_region.astype(np.float32),
        occlusion_gate=occlusion_gate,
        uncertainty_map=uncertainty,
        region_mix=region_mix,
        mask_ref=f"masks/{track_id}.json",
        quality=_quality_label(geometry_quality),
        geometry_quality=geometry_quality,
        field_confidence=field_confidence,
        quality_flags=tuple(sorted(set(flags))),
        metadata={
            "implementation": config.implementation,
            "evidence_frame_id": frame.frame_id,
            "distance_decay_fraction": config.distance_decay_fraction,
            "public_space_constraint": "affected_field_zero_outside_public_space_mask",
        },
    )


class FootprintEstimator:
    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoint = checkpoint or {"config": FootprintConfig().to_dict()}
        self.config = _config_from_checkpoint(self.checkpoint)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "FootprintEstimator":
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(checkpoint)

    def predict(
        self,
        track_id: str,
        tracks: list[DetectorTrackRecord],
        frames: dict[int, FrameRecord],
        segmentation_by_frame: dict[int, SegmentationMaskOutput] | None = None,
    ) -> AffectedRegionFieldOutput:
        return estimate_affected_region_field(track_id, tracks, frames, segmentation_by_frame, self.config)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> FootprintConfig:
    payload = checkpoint.get("config", {})
    return FootprintConfig(
        implementation=str(payload.get("implementation", "deterministic_lamp_conditioned_field_v1")),
        distance_decay_fraction=float(payload.get("distance_decay_fraction", 0.38)),
        downward_gate_strength=float(payload.get("downward_gate_strength", 12.0)),
        occlusion_suppression=float(payload.get("occlusion_suppression", 0.78)),
        weak_geometry_floor=float(payload.get("weak_geometry_floor", 0.48)),
    )


def estimate_footprint(
    track_id: str,
    tracks: list[DetectorTrackRecord],
    frames: dict[int, FrameRecord],
    segmentation_by_frame: dict[int, SegmentationMaskOutput] | None = None,
) -> FootprintEstimate:
    field = estimate_affected_region_field(track_id, tracks, frames, segmentation_by_frame)
    return FootprintEstimate(quality=field.quality, mask_ref=field.mask_ref, geometry_quality=field.geometry_quality, field=field)
