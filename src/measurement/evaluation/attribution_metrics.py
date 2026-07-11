from __future__ import annotations


def attribution_accuracy(pred: list[str], truth: list[str]) -> float:
    if len(pred) != len(truth):
        raise ValueError("pred and truth must have the same length")
    if not pred:
        return 0.0
    return sum(a == b for a, b in zip(pred, truth)) / len(pred)
