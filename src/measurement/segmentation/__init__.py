"""Public-space and confounder segmentation interfaces."""

from rbccps_measurement.segmentation.illumination_disentangled import (
    CONFOUNDER_CLASSES,
    OCCLUDER_CLASSES,
    PUBLIC_SPACE_CLASSES,
    SEMANTIC_CLASSES,
    SegmentationConfig,
    SegmentationEstimator,
    deterministic_segment_frame,
)

__all__ = [
    "CONFOUNDER_CLASSES",
    "OCCLUDER_CLASSES",
    "PUBLIC_SPACE_CLASSES",
    "SEMANTIC_CLASSES",
    "SegmentationConfig",
    "SegmentationEstimator",
    "deterministic_segment_frame",
]
