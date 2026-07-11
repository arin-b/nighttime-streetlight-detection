from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ACCEPTED_BBOX_FORMATS = {"pixel_xyxy_original_frame", "normalized_xyxy_original_frame"}
LEGACY_STREETLIGHT_CLASS = "streetlight"
LAMP_HEAD_CLASS = "streetlight_lamp_head"
POLE_CLASS = "streetlight_pole"
ACCEPTED_TRACK_CLASSES = {LEGACY_STREETLIGHT_CLASS, LAMP_HEAD_CLASS, POLE_CLASS}
MEASUREMENT_SOURCE_CLASSES = {LEGACY_STREETLIGHT_CLASS, LAMP_HEAD_CLASS}


def normalize_track_class(value: str | None) -> str:
    if value in {None, "", LEGACY_STREETLIGHT_CLASS}:
        return LAMP_HEAD_CLASS
    return str(value)


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{context} is missing required keys: {', '.join(missing)}")


@dataclass(frozen=True)
class CameraMetadata:
    exposure_time_s: float | None = None
    sensor_sensitivity_iso: int | None = None
    aperture_f_number: float | None = None
    focal_length_mm: float | None = None
    white_balance_mode: str | None = None
    ae_mode: str | None = None
    awb_mode: str | None = None
    hdr_mode: str | None = None
    night_mode: bool | None = None
    digital_zoom: float | None = None
    ois_eis_enabled: bool | None = None
    metadata_quality: str = "missing"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CameraMetadata":
        if not payload:
            return cls()
        return cls(
            exposure_time_s=payload.get("exposure_time_s"),
            sensor_sensitivity_iso=payload.get("sensor_sensitivity_iso"),
            aperture_f_number=payload.get("aperture_f_number"),
            focal_length_mm=payload.get("focal_length_mm"),
            white_balance_mode=payload.get("white_balance_mode"),
            ae_mode=payload.get("ae_mode"),
            awb_mode=payload.get("awb_mode"),
            hdr_mode=payload.get("hdr_mode"),
            night_mode=payload.get("night_mode"),
            digital_zoom=payload.get("digital_zoom"),
            ois_eis_enabled=payload.get("ois_eis_enabled"),
            metadata_quality=payload.get("metadata_quality", "partial"),
        )

    @property
    def auto_exposure_active(self) -> bool:
        return str(self.ae_mode or "").lower() in {"auto", "on", "continuous"}


@dataclass(frozen=True)
class PoseRecord:
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    gps_accuracy_m: float | None = None
    speed_mps: float | None = None
    heading_deg: float | None = None
    orientation_quat: tuple[float, float, float, float] | None = None
    imu_quality: str = "missing"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PoseRecord":
        if not payload:
            return cls()
        quat = payload.get("orientation_quat")
        return cls(
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude"),
            altitude_m=payload.get("altitude_m"),
            gps_accuracy_m=payload.get("gps_accuracy_m"),
            speed_mps=payload.get("speed_mps"),
            heading_deg=payload.get("heading_deg"),
            orientation_quat=tuple(quat) if quat and len(quat) == 4 else None,
            imu_quality=payload.get("imu_quality", "partial"),
        )


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_ns: int
    image_uri: str
    image_format: str
    width: int
    height: int
    camera: CameraMetadata
    pose: PoseRecord

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameRecord":
        _require_keys(payload, ("frame_id", "timestamp_ns", "image_uri", "width", "height"), "frame")
        return cls(
            frame_id=int(payload["frame_id"]),
            timestamp_ns=int(payload["timestamp_ns"]),
            image_uri=str(payload["image_uri"]),
            image_format=str(payload.get("image_format", "unknown")),
            width=int(payload["width"]),
            height=int(payload["height"]),
            camera=CameraMetadata.from_dict(payload.get("camera")),
            pose=PoseRecord.from_dict(payload.get("pose")),
        )


@dataclass(frozen=True)
class DetectorTrackRecord:
    frame_id: int
    timestamp_ns: int
    track_id: str
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    bbox_format: str
    detector_score: float
    track_confidence: float | None = None
    track_age: int | None = None
    lost_count: int | None = None
    source_model: str | None = None
    optional_cue_scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectorTrackRecord":
        _require_keys(payload, ("frame_id", "timestamp_ns", "track_id", "class_name", "bbox_xyxy", "detector_score"), "track")
        bbox = payload["bbox_xyxy"]
        if len(bbox) != 4:
            raise ValueError(f"track {payload.get('track_id')} bbox_xyxy must have 4 values")
        return cls(
            frame_id=int(payload["frame_id"]),
            timestamp_ns=int(payload["timestamp_ns"]),
            track_id=str(payload["track_id"]),
            class_name=normalize_track_class(str(payload["class_name"])),
            bbox_xyxy=tuple(float(v) for v in bbox),
            bbox_format=str(payload.get("bbox_format", "pixel_xyxy_original_frame")),
            detector_score=float(payload["detector_score"]),
            track_confidence=float(payload["track_confidence"]) if payload.get("track_confidence") is not None else None,
            track_age=int(payload["track_age"]) if payload.get("track_age") is not None else None,
            lost_count=int(payload["lost_count"]) if payload.get("lost_count") is not None else None,
            source_model=payload.get("source_model"),
            optional_cue_scores={str(k): float(v) for k, v in payload.get("optional_cue_scores", {}).items()},
        )


@dataclass(frozen=True)
class CalibrationRecord:
    intrinsics: dict[str, Any] | None = None
    distortion: list[float] | None = None
    mount: dict[str, Any] | None = None
    photometric: dict[str, Any] = field(default_factory=dict)
    map_priors: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CalibrationRecord":
        if not payload:
            return cls()
        return cls(
            intrinsics=payload.get("intrinsics"),
            distortion=payload.get("distortion"),
            mount=payload.get("mount"),
            photometric=payload.get("photometric", {}),
            map_priors=payload.get("map_priors", {}),
        )

    @property
    def has_field_lux_calibration(self) -> bool:
        return bool(self.photometric.get("field_lux_calibration_id"))


@dataclass(frozen=True)
class ClipManifest:
    clip_id: str
    device_id: str
    calibration_level: int
    policy_id: str
    frames: tuple[FrameRecord, ...]
    tracks: tuple[DetectorTrackRecord, ...]
    optional_calibration: CalibrationRecord = field(default_factory=CalibrationRecord)
    video_uri: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClipManifest":
        _require_keys(payload, ("clip_id", "device_id", "calibration_level", "policy_id", "frames", "tracks"), "clip manifest")
        return cls(
            clip_id=str(payload["clip_id"]),
            device_id=str(payload["device_id"]),
            calibration_level=int(payload["calibration_level"]),
            policy_id=str(payload["policy_id"]),
            frames=tuple(FrameRecord.from_dict(item) for item in payload["frames"]),
            tracks=tuple(DetectorTrackRecord.from_dict(item) for item in payload["tracks"]),
            optional_calibration=CalibrationRecord.from_dict(payload.get("optional_calibration")),
            video_uri=payload.get("video_uri"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClipManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls.from_dict(payload)

    def frame_by_id(self) -> dict[int, FrameRecord]:
        return {frame.frame_id: frame for frame in self.frames}
