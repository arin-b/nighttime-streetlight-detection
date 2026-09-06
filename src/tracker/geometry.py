"""Small geometry helpers used by the custom tracker and diagnostics."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


EPSILON = 1e-6


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box[:4])
    return [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0]


def xyxy_to_centre_scale(box: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``[x1, y1, x2, y2]`` to centre and strictly positive size."""

    x1, y1, x2, y2 = (float(value) for value in box[:4])
    return (
        np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float),
        np.asarray([x2 - x1, y2 - y1], dtype=float),
    )


def xywh_to_centre_scale(box: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = (float(value) for value in box[:4])
    return np.asarray([x, y], dtype=float), np.asarray([width, height], dtype=float)


def is_valid_xywh(box: Sequence[float]) -> bool:
    if len(box) < 4:
        return False
    values = [float(value) for value in box[:4]]
    return all(math.isfinite(value) for value in values) and values[2] > 0.0 and values[3] > 0.0


def centre_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def logarithmic_scale_distance(first: np.ndarray, second: np.ndarray) -> float:
    """L2 distance of logarithmic relative width and height changes."""

    first_safe = np.maximum(np.asarray(first, dtype=float), EPSILON)
    second_safe = np.maximum(np.asarray(second, dtype=float), EPSILON)
    return float(np.linalg.norm(np.log(first_safe / second_safe)))


def iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    """IoU for the offline association diagnostic only, never for tracking."""

    ax1, ay1, ax2, ay2 = (float(value) for value in first[:4])
    bx1, by1, bx2, by2 = (float(value) for value in second[:4])
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, EPSILON)
