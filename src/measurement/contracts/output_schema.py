from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeasurementReport:
    measurement_run_id: str
    lamp_observation_id: str
    lamp_track_id: str
    mapped_lamp_id: str | None
    clip_id: str
    time_window: dict[str, Any]
    geo_summary: dict[str, Any]
    status: dict[str, Any]
    affected_region: dict[str, Any]
    metrics: dict[str, Any]
    confidence: dict[str, Any]
    uncertainty_flags: list[str]
    optional_physical_estimates: dict[str, Any]
    traceability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_reports_json(path: str | Path, reports: list[MeasurementReport]) -> None:
    Path(path).write_text(json.dumps([report.to_dict() for report in reports], indent=2), encoding="utf-8")
