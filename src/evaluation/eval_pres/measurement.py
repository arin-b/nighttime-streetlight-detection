"""Brightness measurement engine (PDF §5).

For each tracked streetlight ROI, measure brightness metrics and classify
the lamp as working / off / flickering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from evaluation.eval_pres.audit_config import MeasurementConfig


@dataclass
class LampMeasurement:
    """Single-frame brightness measurement for one tracked lamp."""

    track_id: str
    frame_index: int
    # Bounding box (x1, y1, x2, y2) in pixel coordinates
    xyxy: list[float]
    # Raw metrics
    mean_brightness: float = 0.0
    peak_brightness: float = 0.0  # 95th percentile
    total_flux: float = 0.0  # sum of pixel values
    std_dev: float = 0.0
    max_pixel: float = 0.0
    # Classification for this single frame
    is_on: bool = False
    # Detection confidence from YOLO
    detection_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "xyxy": self.xyxy,
            "mean_brightness": round(self.mean_brightness, 2),
            "peak_brightness": round(self.peak_brightness, 2),
            "total_flux": round(self.total_flux, 2),
            "std_dev": round(self.std_dev, 2),
            "max_pixel": round(self.max_pixel, 2),
            "is_on": self.is_on,
            "detection_confidence": round(self.detection_confidence, 4),
        }


def _apply_gamma(values: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction: linearise sRGB by raising to *gamma* power.

    values should be in [0, 255]. Output is also in [0, 255] range.
    """
    if gamma == 1.0:
        return values.astype(np.float32)
    normalised = values.astype(np.float64) / 255.0
    corrected = np.power(normalised, gamma) * 255.0
    return corrected.astype(np.float32)


def measure_lamp(
    frame: np.ndarray,
    track_id: str,
    frame_index: int,
    xyxy: list[float],
    confidence: float,
    cfg: MeasurementConfig,
) -> LampMeasurement:
    """Measure brightness of a single lamp ROI in one frame.

    Parameters
    ----------
    frame : np.ndarray
        The full video frame (BGR, uint8).
    track_id : str
        Persistent track identifier.
    frame_index : int
        1-based frame index in the video.
    xyxy : list[float]
        Bounding box [x1, y1, x2, y2] in pixel coordinates.
    confidence : float
        YOLO detection confidence for this box.
    cfg : MeasurementConfig
        Configuration knobs.

    Returns
    -------
    LampMeasurement
        Per-frame brightness measurements and on/off classification.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(w, int(xyxy[2]))
    y2 = min(h, int(xyxy[3]))

    measurement = LampMeasurement(
        track_id=track_id,
        frame_index=frame_index,
        xyxy=xyxy,
        detection_confidence=confidence,
    )

    if x2 <= x1 or y2 <= y1:
        return measurement

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return measurement

    # Extract brightness channel
    if cfg.use_hsv_v_channel:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        channel = hsv[:, :, 2].astype(np.float32)
    else:
        channel = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Optional gamma correction (sRGB linearisation)
    if cfg.gamma_correction != 1.0:
        channel = _apply_gamma(channel.astype(np.uint8), cfg.gamma_correction)

    # Compute metrics
    measurement.mean_brightness = float(np.mean(channel))
    measurement.peak_brightness = float(np.percentile(channel, 95))
    measurement.total_flux = float(np.sum(channel))
    measurement.std_dev = float(np.std(channel))
    measurement.max_pixel = float(np.max(channel))

    # On/Off classification
    measurement.is_on = measurement.mean_brightness >= cfg.brightness_threshold

    return measurement
