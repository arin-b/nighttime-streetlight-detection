from __future__ import annotations


def summarize_errors(false_positives: int, false_negatives: int) -> dict[str, int]:
    return {"false_positives": false_positives, "false_negatives": false_negatives}
