from __future__ import annotations


def brier_score(probabilities: list[float], labels: list[int]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)
