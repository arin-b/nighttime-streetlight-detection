from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(report["metrics"]["overall_category"] for report in reports)
    actions = Counter(report["confidence"].get("action", "report") for report in reports)
    physical_valid = sum(1 for report in reports if report["optional_physical_estimates"]["valid"])
    if reports:
        mean_confidence = sum(float(report["confidence"]["overall"]) for report in reports) / len(reports)
    else:
        mean_confidence = 0.0
    return {
        "num_reports": len(reports),
        "categories": dict(categories),
        "actions": dict(actions),
        "physical_valid_reports": physical_valid,
        "mean_confidence": round(mean_confidence, 4),
    }
