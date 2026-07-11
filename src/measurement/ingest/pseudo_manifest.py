from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.ingest.validation import validate_clip_manifest


@dataclass(frozen=True)
class PseudoManifestOptions:
    clip_id: str
    device_id: str = "pseudo_android_phone"
    fps: float = 30.0
    start_timestamp_ns: int = 1_700_000_000_000_000_000
    calibration_level: int = 1
    policy_id: str = "rbccps_measurement_policy_v1"
    max_lamps_per_frame: int = 2
    latitude: float | None = None
    longitude: float | None = None
    heading_deg: float | None = None
    copy_images: bool = False


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            boxes.append((min_x, min_y, max_x + 1, max_y + 1, area))
    return boxes


def _estimate_lamp_boxes(path: Path, max_lamps: int) -> list[tuple[float, float, float, float, float]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original_width, original_height = rgb.size
        scale = min(1.0, 512.0 / max(original_width, original_height))
        small = rgb.resize((max(1, int(original_width * scale)), max(1, int(original_height * scale))))

    arr = np.asarray(small, dtype=np.float32)
    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    upper_scene = np.zeros_like(luma, dtype=bool)
    upper_scene[: max(1, int(luma.shape[0] * 0.72)), :] = True
    threshold = max(170.0, float(np.percentile(luma[upper_scene], 99.4)))
    mask = (luma >= threshold) & upper_scene

    candidates = []
    for x1, y1, x2, y2, area in _component_boxes(mask):
        box_width = x2 - x1
        box_height = y2 - y1
        if area < 3 or box_width > mask.shape[1] * 0.2 or box_height > mask.shape[0] * 0.2:
            continue
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        score = float(luma[y1:y2, x1:x2].mean() / 255.0)
        pad = max(4, int(max(box_width, box_height) * 1.8))
        ox1 = max(0.0, (cx - pad) / scale)
        oy1 = max(0.0, (cy - pad) / scale)
        ox2 = min(float(original_width), (cx + pad) / scale)
        oy2 = min(float(original_height), (cy + pad) / scale)
        candidates.append((ox1, oy1, ox2, oy2, score, area))

    candidates.sort(key=lambda item: (item[5], item[4]), reverse=True)
    boxes = [(x1, y1, x2, y2, score) for x1, y1, x2, y2, score, _ in candidates[:max_lamps]]
    if boxes:
        return boxes

    fallback_width = max(24.0, original_width * 0.035)
    fallback_height = max(36.0, original_height * 0.07)
    cx = original_width * 0.5
    cy = original_height * 0.22
    return [(
        max(0.0, cx - fallback_width),
        max(0.0, cy - fallback_height),
        min(float(original_width), cx + fallback_width),
        min(float(original_height), cy + fallback_height),
        0.25,
    )]


def build_pseudo_manifest(image_paths: list[Path], output_manifest: Path, options: PseudoManifestOptions) -> ClipManifest:
    if not image_paths:
        raise ValueError("at least one image is required")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame_root = output_manifest.parent / "frames"
    if options.copy_images:
        frame_root.mkdir(parents=True, exist_ok=True)

    frame_interval_ns = int(1_000_000_000 / options.fps)
    frames = []
    tracks = []
    for index, source in enumerate(image_paths, start=1):
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"image does not exist: {source}")
        target = source
        if options.copy_images:
            suffix = source.suffix or ".jpg"
            target = frame_root / f"{index:06d}{suffix.lower()}"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)

        width, height = _image_size(target)
        timestamp_ns = options.start_timestamp_ns + (index - 1) * frame_interval_ns
        frames.append({
            "frame_id": index,
            "timestamp_ns": timestamp_ns,
            "image_uri": _relative_or_absolute(target, output_manifest.parent),
            "image_format": target.suffix.lower().lstrip(".") or "unknown",
            "width": width,
            "height": height,
            "camera": {
                "exposure_time_s": round(1.0 / max(options.fps, 1.0), 6),
                "sensor_sensitivity_iso": 800,
                "ae_mode": "auto",
                "hdr_mode": "unknown",
                "night_mode": True,
                "metadata_quality": "pseudo",
            },
            "pose": {
                "latitude": options.latitude,
                "longitude": options.longitude,
                "gps_accuracy_m": None if options.latitude is None or options.longitude is None else 15.0,
                "heading_deg": options.heading_deg,
                "imu_quality": "pseudo",
            },
        })

        boxes = _estimate_lamp_boxes(target, options.max_lamps_per_frame)
        for lamp_index, (x1, y1, x2, y2, score) in enumerate(boxes, start=1):
            tracks.append({
                "frame_id": index,
                "timestamp_ns": timestamp_ns,
                "track_id": f"pseudo_lamp_{lamp_index}",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_format": "pixel_xyxy_original_frame",
                "detector_score": round(max(0.2, min(0.95, score)), 4),
                "track_confidence": round(max(0.2, min(0.85, score * 0.85)), 4),
                "track_age": len(image_paths),
                "lost_count": 0,
                "source_model": "pseudo_bright_region_estimator_v1",
                "optional_cue_scores": {
                    "pseudo_generated": 1.0,
                    "bright_region_score": round(score, 4),
                },
            })

    payload = {
        "clip_id": options.clip_id,
        "device_id": options.device_id,
        "calibration_level": options.calibration_level,
        "policy_id": options.policy_id,
        "video_uri": None,
        "frames": frames,
        "tracks": tracks,
        "optional_calibration": {
            "photometric": {"field_lux_calibration_id": None},
            "map_priors": {},
        },
    }
    manifest = ClipManifest.from_dict(payload)
    validate_clip_manifest(manifest)
    output_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest
