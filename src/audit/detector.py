"""YOLO26 detection module (PDF §2).

Wraps the Ultralytics YOLO API for frame-by-frame inference and
video-level tracking.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

from audit_pipeline.config import DetectorConfig, TrackerConfig


_NON_STREETLIGHT_LABELS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "vehicle",
}

_ONE_CLASS_STREETLIGHT_ALIASES = {
    "0",
    "object",
    "streetlight",
    "street light",
    "street lamp",
    "streetlamp",
    "lamp post",
    "lamppost",
}


def _import_yolo():
    """Lazy-import ultralytics, setting config dir to avoid polluting home."""
    config_dir = Path(__file__).resolve().parents[1] / "_ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    from ultralytics import YOLO
    return YOLO


def load_model(cfg: DetectorConfig):
    """Load a YOLO model from the configured checkpoint path."""
    model_path = Path(cfg.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Please set --model to the path of your fine-tuned YOLO weights."
        )
    if cfg.use_geometry_attention or cfg.use_cse or cfg.use_negative_attention:
        from rbccps_od.models.yolo_ablation import build_yolo26_ablation_model

        return build_yolo26_ablation_model(
            model_path,
            use_geometry_attention=cfg.use_geometry_attention,
            use_cse=cfg.use_cse,
            use_negative_attention=cfg.use_negative_attention,
            negative_mask_loss_weight=cfg.negative_mask_loss_weight,
        )
    YOLO = _import_yolo()
    return YOLO(str(model_path))


def _normalise_label(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").replace("-", " ").split())


def _compact_label(label: str) -> str:
    return _normalise_label(label).replace(" ", "")


def get_model_class_names(model: Any) -> dict[int, str]:
    """Return model class names as ``{class_id: name}``."""
    names = getattr(model, "names", None)
    if names is None and getattr(model, "model", None) is not None:
        names = getattr(model.model, "names", None)

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _format_class_names(class_names: dict[int, str], limit: int = 12) -> str:
    items = [f"{idx}: {name}" for idx, name in sorted(class_names.items())[:limit]]
    if len(class_names) > limit:
        items.append(f"... {len(class_names) - limit} more")
    return ", ".join(items) if items else "no class names found"


def _label_matches_target(label: str, target_labels: list[str]) -> bool:
    label_norm = _normalise_label(label)
    label_compact = _compact_label(label)
    for target in target_labels:
        target_norm = _normalise_label(target)
        if label_norm == target_norm or label_compact == target_norm.replace(" ", ""):
            return True
    return False


def _is_single_class_streetlight_model(class_names: dict[int, str]) -> bool:
    if len(class_names) != 1:
        return False
    only_name = _normalise_label(next(iter(class_names.values())))
    return only_name in _ONE_CLASS_STREETLIGHT_ALIASES


def _looks_like_non_streetlight(label: str) -> bool:
    return _normalise_label(label) in _NON_STREETLIGHT_LABELS


def resolve_target_classes(model: Any, cfg: DetectorConfig) -> list[int]:
    """Resolve and validate the class IDs that should be treated as lamps.

    A one-class fine-tuned detector is accepted even when its class is named
    ``0`` or ``object``. Multi-class models must expose a streetlight-like
    class name unless the user supplies a valid streetlight class ID.
    """
    class_names = get_model_class_names(model)

    if cfg.target_classes is not None:
        selected = [int(class_id) for class_id in cfg.target_classes]
        cfg.target_classes = selected
        cfg.resolved_class_names = {
            class_id: class_names.get(class_id, str(class_id)) for class_id in selected
        }
        if class_names:
            bad = [
                f"{class_id}: {class_names[class_id]}"
                for class_id in selected
                if class_id in class_names
                and not _label_matches_target(class_names[class_id], cfg.target_labels)
                and not _is_single_class_streetlight_model(class_names)
            ]
            if bad:
                raise ValueError(
                    "Selected class IDs do not look like streetlight classes: "
                    f"{', '.join(bad)}. The model exposes: "
                    f"{_format_class_names(class_names)}. Use a fine-tuned "
                    "streetlight checkpoint or pass the correct streetlight class ID."
                )
        return selected

    matches = [
        class_id
        for class_id, name in sorted(class_names.items())
        if _label_matches_target(name, cfg.target_labels)
    ]

    if not matches and _is_single_class_streetlight_model(class_names):
        matches = [next(iter(class_names))]

    if not matches:
        if any(_looks_like_non_streetlight(name) for name in class_names.values()):
            raise ValueError(
                "This checkpoint appears to use generic COCO-style classes, not a "
                "streetlight class. For example: "
                f"{_format_class_names(class_names)}. The audit pipeline now refuses "
                "to treat class 0 as streetlight because class 0 is often 'person'. "
                "Use the fine-tuned streetlight weights, or pass --classes with the "
                "actual streetlight class ID if your model has one."
            )
        raise ValueError(
            "Could not find a streetlight class in the model names. Looked for: "
            f"{', '.join(cfg.target_labels)}. Model exposes: "
            f"{_format_class_names(class_names)}."
        )

    cfg.target_classes = matches
    cfg.resolved_class_names = {
        class_id: class_names.get(class_id, str(class_id)) for class_id in matches
    }
    return matches


def stream_video_with_tracking(
    model: Any,
    video_path: str | Path,
    detector_cfg: DetectorConfig,
    tracker_yaml_path: str | Path,
    tracker_cfg: TrackerConfig,
) -> Iterator[Any]:
    """Yield per-frame YOLO results with persistent tracking IDs.

    Uses ``model.track(stream=True, persist=True)`` so that each result has
    ``boxes.id`` populated with BoT-SORT or ByteTrack track IDs.
    """
    results = model.track(
        source=str(video_path),
        stream=True,
        conf=detector_cfg.conf_threshold,
        iou=detector_cfg.iou_threshold,
        imgsz=detector_cfg.imgsz,
        device=detector_cfg.device,
        classes=detector_cfg.target_classes,  # only detect streetlight class(es)
        tracker=str(tracker_yaml_path),
        persist=True,
        vid_stride=tracker_cfg.vid_stride,
        verbose=False,
        save=False,
    )
    try:
        for result in results:
            # Coerce list results (some YOLO versions return a list per frame)
            if isinstance(result, (list, tuple)):
                result = result[0] if result else None
            yield result
    except Exception as exc:
        # Ultralytics trackers may encounter unstable Kalman updates in some cases.
        # Fall back to detection-only streaming so the pipeline can continue.
        from numpy.linalg import LinAlgError

        if isinstance(exc, LinAlgError):
            print(
                "WARNING: Tracker failed due to unstable Kalman filter; "
                "falling back to detection-only inference.",
                file=sys.stderr,
            )
            try:
                if hasattr(results, "close"):
                    results.close()
            except Exception:
                pass

            fallback_results = model.predict(
                source=str(video_path),
                stream=True,
                conf=detector_cfg.conf_threshold,
                iou=detector_cfg.iou_threshold,
                imgsz=detector_cfg.imgsz,
                device=detector_cfg.device,
                classes=detector_cfg.target_classes,
                verbose=False,
                save=False,
                vid_stride=tracker_cfg.vid_stride,
            )
            for result in fallback_results:
                if isinstance(result, (list, tuple)):
                    result = result[0] if result else None
                yield result
            if hasattr(fallback_results, "close"):
                try:
                    fallback_results.close()
                except Exception:
                    pass
        else:
            raise
    finally:
        if hasattr(results, "close"):
            try:
                results.close()
            except Exception:
                pass
