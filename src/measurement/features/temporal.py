from __future__ import annotations


def stability_from_jitter(jitter: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(jitter)))
