from __future__ import annotations


def flip_rate(labels: list[str]) -> float:
    if len(labels) < 2:
        return 0.0
    flips = sum(a != b for a, b in zip(labels, labels[1:]))
    return flips / (len(labels) - 1)
