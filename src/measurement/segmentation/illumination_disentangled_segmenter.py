from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationBackboneSpec:
    pretrained_asset: str = "segmentation_deeplabv3_mobilenet_v3"
    enhanced_view_policy: str = "auxiliary_only"
    measurement_source: str = "original_or_normalized_capture"
