from __future__ import annotations


def occlusion_penalty(occluded_probability: float) -> float:
    return max(0.0, min(1.0, float(occluded_probability)))
