"""Pipeline configuration with all tunable parameters.

Every knob mentioned in the PDF proposal is exposed here as a dataclass field
so that the user can override from the CLI or by editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_STREETLIGHT_TARGET_LABELS = (
    "streetlight",
    "street light",
    "street_lamp",
    "street lamp",
    "lamp post",
    "lamppost",
)


@dataclass
class DetectorConfig:
    """YOLO26 detector settings (PDF §2)."""

    model_path: str = "PLACEHOLDER_MODEL_WEIGHTS.pt"
    # Match the video_runner default so the audit pipeline keeps more detections.
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 1280
    device: str = "0"  # GPU index or "cpu"
    # If set, these numeric class IDs are used after validating their model
    # names. If None, the pipeline resolves streetlight classes by name.
    target_classes: list[int] | None = None
    target_labels: list[str] = field(
        default_factory=lambda: list(DEFAULT_STREETLIGHT_TARGET_LABELS)
    )
    use_geometry_attention: bool = False
    use_cse: bool = False
    use_negative_attention: bool = False
    negative_mask_loss_weight: float = 1.0
    # Filled at runtime after the model is loaded.
    resolved_class_names: dict[int, str] = field(default_factory=dict)


@dataclass
class TrackerConfig:
    """Multi-object tracker settings (PDF §3)."""

    tracker_type: str = "botsort"  # "botsort" or "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    gmc_method: str = "sparseOptFlow"
    with_reid: bool = False
    vid_stride: int = 1


@dataclass
class MultiCueConfig:
    """Multi-cue detection filtering (PDF §4)."""

    enabled: bool = True
    aggregation_threshold: float = 0.5
    # Aspect-ratio filter: w/h must be in [min, max]
    aspect_ratio_min: float = 0.3
    aspect_ratio_max: float = 3.0
    # Spatial filter: reject detections whose center_y > fraction * frame_height
    # Relaxed to be less aggressive than the previous default.
    spatial_upper_fraction: float = 0.95
    # Brightness cue: minimum max-pixel intensity to keep a detection
    min_brightness_for_detection: int = 10
    # Temporal consistency: minimum frames a track must persist
    min_track_frames_for_confirmation: int = 1
    # Duplicate removal: max pixel distance between track centers to merge
    duplicate_center_distance_px: float = 80.0


@dataclass
class MeasurementConfig:
    """Brightness measurement engine (PDF §5)."""

    brightness_threshold: int = 50  # grayscale value for on/off classification
    gamma_correction: float = 1.0  # 1.0 = no correction, 2.2 = sRGB linearisation
    use_hsv_v_channel: bool = False  # if True, use V channel instead of grayscale
    flicker_crossing_threshold: int = 3  # min on/off crossings to flag as flickering


@dataclass
class AggregationConfig:
    """Temporal aggregation settings (PDF §6)."""

    min_track_frames: int = 5  # tracks shorter than this are discarded
    working_frame_fraction: float = 0.75  # fraction of "on" frames to mark as working
    merge_distance_px: float = 60.0  # spatial proximity threshold for merging broken tracks


@dataclass
class EvaluationConfig:
    """Evaluation metrics settings (PDF §7)."""

    gt_labels_dir: str | None = None  # path to ground-truth YOLO label txt files
    gt_status_file: str | None = None  # optional CSV with (track_id, status) ground truth
    iou_thresholds: list[float] = field(
        default_factory=lambda: [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95
    )


@dataclass
class LocationPriorConfig:
    """Device- and route-aware location memory for repeated audits."""

    prior_path: str | None = None
    output_path: str | None = None
    location_samples_path: str | None = None
    capture_device_id: str | None = None
    route_group: str | None = None
    capture_latitude: float | None = None
    capture_longitude: float | None = None
    capture_gps_accuracy_m: float | None = None
    query_latitude: float | None = None
    query_longitude: float | None = None
    query_gps_accuracy_m: float | None = None
    prior_only: bool = False
    match_radius_m: float = 12.0
    good_gps_match_radius_m: float = 8.0
    min_observations_for_existing: int = 2
    min_devices_for_high_confidence: int = 2
    existence_confidence_threshold: float = 0.72


@dataclass
class AuditPipelineConfig:
    """Top-level config aggregating all sub-configs."""

    video_path: str = "PLACEHOLDER_TEST_VIDEO.mp4"
    output_dir: str | None = None  # auto-derived from video name if None
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    multicue: MultiCueConfig = field(default_factory=MultiCueConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    location_prior: LocationPriorConfig = field(default_factory=LocationPriorConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire config for the report."""
        import dataclasses
        return dataclasses.asdict(self)
