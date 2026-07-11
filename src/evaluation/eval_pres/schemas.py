from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoxRecord:
    frame_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float = 1.0
    track_id: str | None = None
    physical_lamp_id: str | None = None
    inventory_id: str | None = None
    status: str | None = None
    illumination_class: str | None = None
    affected_region_polygon: tuple[tuple[float, float], ...] | None = None
    served_area_fraction: float | None = None
    confounder_present: bool | None = None
    target_attribution_correct: bool | None = None
    mapped_lamp_id: str | None = None


@dataclass(frozen=True)
class ReportRecord:
    lamp_observation_id: str
    lamp_track_id: str
    mapped_lamp_id: str | None
    clip_id: str
    status_label: str | None
    overall_category: str | None
    overall_score: float | None
    confidence: float | None
    affected_region_area_fraction: float | None
    affected_region_polygon: tuple[tuple[float, float], ...] | None = None
    attribution_class: str | None = None
    target_attribution_score: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationInputs:
    predictions: tuple[BoxRecord, ...]
    ground_truth: tuple[BoxRecord, ...]
    reports: tuple[ReportRecord, ...]
    route_distance_km: float | None = None
    latency_seconds: float | None = None
    model_size_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult:
    name: str
    section: str
    direction: str
    value: float | int | None
    unit: str
    description: str
    status: str = "computed"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "metric": self.name,
            "direction": self.direction,
            "value": self.value,
            "unit": self.unit,
            "description": self.description,
            "status": self.status,
            "reason": self.reason,
        }

