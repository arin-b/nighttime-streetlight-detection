from __future__ import annotations


def uniformity_from_low_high(low_quantile: float, high_quantile: float) -> float:
    if high_quantile <= 0:
        return 0.0
    return max(0.0, min(1.0, low_quantile / high_quantile))
