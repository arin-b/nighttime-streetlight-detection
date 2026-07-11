from __future__ import annotations


def adequacy_class_from_score(score: float) -> str:
    if score >= 0.72:
        return "adequate"
    if score >= 0.45:
        return "marginal"
    if score >= 0.2:
        return "poor"
    return "unknown"
