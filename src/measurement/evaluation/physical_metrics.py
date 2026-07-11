from __future__ import annotations


def mean_absolute_error(pred: list[float], truth: list[float]) -> float:
    if len(pred) != len(truth):
        raise ValueError("pred and truth must have the same length")
    if not pred:
        return 0.0
    return sum(abs(a - b) for a, b in zip(pred, truth)) / len(pred)
