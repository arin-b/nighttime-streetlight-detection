from __future__ import annotations

import numpy as np


def saturation_flag(value: float, threshold: float = 0.98) -> bool:
    return float(value) >= threshold


def saturation_mask(rgb01: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    return np.max(rgb01, axis=2) >= threshold


def glare_mask(rgb01: np.ndarray, threshold: float = 0.92) -> np.ndarray:
    return np.max(rgb01, axis=2) >= threshold
