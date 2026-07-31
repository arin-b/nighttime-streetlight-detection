from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from detection.config.paths import repo_root
from detection.config.schemas import CueWeights
from evaluation.eval_pres.audit_config import DetectorConfig, MultiCueConfig


DEFAULT_HYPERPARAMETER_PATH = repo_root() / "configs" / "detection" / "hyperparameters.yaml"


def _default_payload() -> dict[str, Any]:
    detector = DetectorConfig()
    multicue = MultiCueConfig()
    return {
        "detector": {
            "model_path": detector.model_path,
            "conf_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold,
            "imgsz": detector.imgsz,
            "device": detector.device,
            "target_classes": detector.target_classes,
            "target_labels": detector.target_labels,
            "use_geometry_attention": detector.use_geometry_attention,
            "use_cse": detector.use_cse,
            "use_negative_attention": detector.use_negative_attention,
            "negative_mask_loss_weight": detector.negative_mask_loss_weight,
            "geometry_attention": {
                "kernel_size": 7,
                "neutral_bias": 4.0,
            },
            "negative_attention": {
                "reduction": 4,
                "init_bias": -4.0,
            },
        },
        "multicue": {
            "enabled": multicue.enabled,
            "aggregation_threshold": multicue.aggregation_threshold,
            "cue_weights": {
                "trajectory": CueWeights().trajectory,
                "size_progression": CueWeights().size_progression,
                "light_characteristics": CueWeights().light_characteristics,
                "position_prior": CueWeights().position_prior,
            },
            "aspect_ratio_min": multicue.aspect_ratio_min,
            "aspect_ratio_max": multicue.aspect_ratio_max,
            "spatial_upper_fraction": multicue.spatial_upper_fraction,
            "min_brightness_for_detection": multicue.min_brightness_for_detection,
            "min_track_frames_for_confirmation": multicue.min_track_frames_for_confirmation,
            "duplicate_center_distance_px": multicue.duplicate_center_distance_px,
        },
    }


def load_detection_hyperparameters(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_HYPERPARAMETER_PATH
    if not config_path.is_absolute():
        config_path = (repo_root() / config_path).resolve()
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    else:
        loaded = {}

    payload = deepcopy(_default_payload())
    if loaded:
        _merge_dict(payload, loaded)
    return payload


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value


def build_detector_config_from_hyperparams(payload: dict[str, Any]) -> DetectorConfig:
    detector_payload = payload.get("detector", {})
    return DetectorConfig(
        model_path=str(detector_payload.get("model_path", "")),
        conf_threshold=float(detector_payload.get("conf_threshold", 0.25)),
        iou_threshold=float(detector_payload.get("iou_threshold", 0.45)),
        imgsz=int(detector_payload.get("imgsz", 1280)),
        device=str(detector_payload.get("device", "0")),
        target_classes=detector_payload.get("target_classes"),
        target_labels=list(detector_payload.get("target_labels", [])),
        use_geometry_attention=bool(detector_payload.get("use_geometry_attention", False)),
        use_cse=bool(detector_payload.get("use_cse", False)),
        use_negative_attention=bool(detector_payload.get("use_negative_attention", False)),
        negative_mask_loss_weight=float(detector_payload.get("negative_mask_loss_weight", 1.0)),
    )


def build_multicue_config_from_hyperparams(payload: dict[str, Any]) -> MultiCueConfig:
    multicue_payload = payload.get("multicue", {})
    cue_weights_payload = multicue_payload.get("cue_weights", {})
    cue_weights = CueWeights(
        trajectory=float(cue_weights_payload.get("trajectory", 0.25)),
        size_progression=float(cue_weights_payload.get("size_progression", 0.25)),
        light_characteristics=float(cue_weights_payload.get("light_characteristics", 0.25)),
        position_prior=float(cue_weights_payload.get("position_prior", 0.25)),
    )
    return MultiCueConfig(
        enabled=bool(multicue_payload.get("enabled", True)),
        aggregation_threshold=float(multicue_payload.get("aggregation_threshold", 0.5)),
        aspect_ratio_min=float(multicue_payload.get("aspect_ratio_min", 0.3)),
        aspect_ratio_max=float(multicue_payload.get("aspect_ratio_max", 3.0)),
        spatial_upper_fraction=float(multicue_payload.get("spatial_upper_fraction", 0.95)),
        min_brightness_for_detection=int(multicue_payload.get("min_brightness_for_detection", 10)),
        min_track_frames_for_confirmation=int(multicue_payload.get("min_track_frames_for_confirmation", 1)),
        duplicate_center_distance_px=float(multicue_payload.get("duplicate_center_distance_px", 80.0)),
        cue_weights=cue_weights,
    )
