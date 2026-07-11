from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelAssetSpec:
    name: str
    version: str
    filename: str
    url: str | None = None
    checksum: str | None = None
    local_path: str | None = None
    implementation: str = "native"
    source_repo_url: str | None = None
    paper_url: str | None = None
    license_name: str | None = None
    description: str = ""
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CueWeights:
    trajectory: float = 0.25
    size_progression: float = 0.25
    light_characteristics: float = 0.25
    position_prior: float = 0.25


@dataclass
class AdvancedPipelineConfig:
    detector_asset: str = "yolov26_base"
    enhancement_asset: str = "zero_dce_epoch99"
    retinex_asset: str = "retinex_decom_9200"
    domain_adaptation_asset: str = "domain_adaptation"
    tracker_asset: str = "tracker_bundle"
    enable_enhancement: bool = False
    enable_paired_input: bool = False
    enable_retinex: bool = False
    enable_domain_adaptation: bool = False
    enable_tracking: bool = False
    enable_multicue: bool = False
    aggregation_threshold: float = 0.5
    cue_weights: CueWeights = field(default_factory=CueWeights)


@dataclass
class DetectionRunConfig:
    image_path: Path | None = None
    image_dir: Path | None = None
    conf: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    dry_run: bool = False
