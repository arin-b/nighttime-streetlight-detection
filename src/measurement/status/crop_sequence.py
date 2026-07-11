from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_measurement.contracts.input_schema import DetectorTrackRecord, FrameRecord
from rbccps_measurement.contracts.module_io import LampCropSequence, NormalizedFrameProduct
from rbccps_measurement.normalization.module1 import CaptureNormalizer


TOKEN_NAMES = (
    "exposure_factor",
    "saturation_fraction",
    "bloom_fraction",
    "glare_fraction",
    "reliability_score",
    "detector_score",
    "track_confidence",
    "lost_count_norm",
    "metadata_quality_score",
)


@dataclass(frozen=True)
class CropSequenceConfig:
    sequence_length: int = 16
    crop_size: int = 64
    padding_ratio: float = 0.35


def bbox_xyxy_pixels(track: DetectorTrackRecord, frame: FrameRecord) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = track.bbox_xyxy
    if track.bbox_format == "normalized_xyxy_original_frame":
        return x1 * frame.width, y1 * frame.height, x2 * frame.width, y2 * frame.height
    return x1, y1, x2, y2


def padded_bbox_xyxy(
    track: DetectorTrackRecord,
    frame: FrameRecord,
    padding_ratio: float = 0.35,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy_pixels(track, frame)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad = padding_ratio * max(width, height)
    px1 = int(np.floor(max(0.0, x1 - pad)))
    py1 = int(np.floor(max(0.0, y1 - pad)))
    px2 = int(np.ceil(min(float(frame.width), x2 + pad)))
    py2 = int(np.ceil(min(float(frame.height), y2 + pad)))
    if px2 <= px1:
        px2 = min(frame.width, px1 + 1)
    if py2 <= py1:
        py2 = min(frame.height, py1 + 1)
    return px1, py1, px2, py2


def _resolve_image_path(frame: FrameRecord, frame_root: str | Path) -> Path:
    image_path = Path(frame.image_uri)
    return image_path if image_path.is_absolute() else Path(frame_root) / image_path


def _metadata_quality_score(value: str) -> float:
    return {
        "complete": 1.0,
        "controlled": 1.0,
        "good": 0.9,
        "partial": 0.65,
        "pseudo": 0.45,
        "missing": 0.2,
        "poor": 0.15,
    }.get(str(value or "").lower(), 0.35)


def _crop_rgb01(image_path: Path, bbox: tuple[int, int, int, int], crop_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        crop = image.convert("RGB").crop(bbox).resize((crop_size, crop_size), Image.Resampling.BILINEAR)
        return np.asarray(crop, dtype=np.float32) / 255.0


def _token_from_product(track: DetectorTrackRecord, frame: FrameRecord, product: NormalizedFrameProduct | None) -> np.ndarray:
    if product is None:
        exposure = frame.camera.exposure_time_s or 0.0167
        iso = frame.camera.sensor_sensitivity_iso or 800
        exposure_factor = max(0.05, min(64.0, (exposure / 0.0167) * (iso / 800.0)))
        saturation = bloom = glare = 0.0
        reliability = _metadata_quality_score(frame.camera.metadata_quality)
    else:
        exposure_factor = product.exposure_factor
        saturation = float(np.mean(product.saturation_mask))
        bloom = float(np.mean(product.bloom_mask))
        glare = float(np.mean(product.glare_mask))
        reliability = product.reliability_score
    return np.asarray(
        [
            float(exposure_factor),
            saturation,
            bloom,
            glare,
            float(reliability),
            float(track.detector_score),
            float(track.track_confidence if track.track_confidence is not None else track.detector_score),
            min(1.0, float(track.lost_count or 0) / 8.0),
            _metadata_quality_score(frame.camera.metadata_quality),
        ],
        dtype=np.float32,
    )


def _select_window(records: list[DetectorTrackRecord], sequence_length: int) -> list[DetectorTrackRecord]:
    ordered = sorted(records, key=lambda item: (item.timestamp_ns, item.frame_id))
    return ordered[-sequence_length:]


def build_lamp_crop_sequence(
    track_id: str,
    track_records: list[DetectorTrackRecord],
    frames: dict[int, FrameRecord],
    frame_root: str | Path,
    normalized_products: dict[int, NormalizedFrameProduct] | None = None,
    normalizer: CaptureNormalizer | None = None,
    config: CropSequenceConfig | None = None,
) -> LampCropSequence:
    config = config or CropSequenceConfig()
    normalizer = normalizer or CaptureNormalizer()
    normalized_products = normalized_products or {}
    selected = _select_window(track_records, config.sequence_length)

    crop_tensor = np.zeros((config.sequence_length, config.crop_size, config.crop_size, 3), dtype=np.float32)
    valid_mask = np.zeros((config.sequence_length,), dtype=bool)
    bboxes = np.zeros((config.sequence_length, 4), dtype=np.float32)
    tokens = np.zeros((config.sequence_length, len(TOKEN_NAMES)), dtype=np.float32)
    frame_ids: list[int] = [0] * config.sequence_length
    timestamps: list[int] = [0] * config.sequence_length
    flags: list[str] = []

    offset = config.sequence_length - len(selected)
    for index, track in enumerate(selected, start=offset):
        frame = frames[track.frame_id]
        frame_ids[index] = frame.frame_id
        timestamps[index] = frame.timestamp_ns
        bbox = padded_bbox_xyxy(track, frame, config.padding_ratio)
        bboxes[index] = np.asarray(bbox, dtype=np.float32)
        image_path = _resolve_image_path(frame, frame_root)
        product = normalized_products.get(frame.frame_id)
        if image_path.exists():
            crop_tensor[index] = _crop_rgb01(image_path, bbox, config.crop_size)
            if product is None:
                try:
                    product = normalizer.normalize_path(image_path, frame)
                except Exception:
                    flags.append("normalization_product_unavailable")
        else:
            flags.append("image_missing")
        tokens[index] = _token_from_product(track, frame, product)
        valid_mask[index] = True

    if not selected:
        flags.append("empty_track_window")
    if len(selected) < config.sequence_length:
        flags.append("sequence_padded")

    return LampCropSequence(
        track_id=track_id,
        crop_tensor=crop_tensor,
        valid_mask=valid_mask,
        frame_ids=tuple(frame_ids),
        timestamps_ns=tuple(timestamps),
        bbox_xyxy=bboxes,
        metadata_tokens=tokens,
        token_names=TOKEN_NAMES,
        quality_flags=tuple(sorted(set(flags))),
    )


def sequence_to_jsonable(sequence: LampCropSequence) -> dict[str, object]:
    return {
        "track_id": sequence.track_id,
        "frame_ids": list(sequence.frame_ids),
        "timestamps_ns": list(sequence.timestamps_ns),
        "bbox_xyxy": json.loads(json.dumps(sequence.bbox_xyxy.tolist())),
        "metadata_tokens": json.loads(json.dumps(sequence.metadata_tokens.tolist())),
        "token_names": list(sequence.token_names),
        "valid_mask": sequence.valid_mask.astype(int).tolist(),
        "quality_flags": list(sequence.quality_flags),
    }
