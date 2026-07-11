from __future__ import annotations


def glare_penalty_from_saturation(saturated_fraction: float) -> float:
    return max(0.0, min(1.0, 0.08 + 0.92 * float(saturated_fraction)))
