from __future__ import annotations

import numpy as np

from rbccps_measurement.normalization.module1 import srgb_inverse_luma


def monotonic_identity_inverse(value: float) -> float:
    """Placeholder monotonic inverse until a learned response curve is trained."""

    return max(0.0, float(value))


def monotonic_srgb_inverse(rgb01: np.ndarray) -> np.ndarray:
    """Deterministic monotonic inverse response used by Module 1 before training."""

    return srgb_inverse_luma(rgb01)
