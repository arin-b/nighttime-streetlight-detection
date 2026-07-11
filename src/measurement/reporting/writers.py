from __future__ import annotations

import csv
import json
from pathlib import Path

from rbccps_measurement.contracts.output_schema import MeasurementReport


def write_json(path: str | Path, reports: list[MeasurementReport]) -> None:
    Path(path).write_text(json.dumps([report.to_dict() for report in reports], indent=2), encoding="utf-8")


def write_csv(path: str | Path, reports: list[MeasurementReport]) -> None:
    rows = []
    for report in reports:
        rows.append({
            "lamp_observation_id": report.lamp_observation_id,
            "lamp_track_id": report.lamp_track_id,
            "clip_id": report.clip_id,
            "status": report.status["label"],
            "overall_category": report.metrics["overall_category"],
            "overall_score": report.metrics["overall_useful_illumination_score"],
            "confidence": report.confidence["overall"],
            "attribution_class": report.confidence["attribution_class"],
            "physical_valid": report.optional_physical_estimates["valid"],
        })
    fieldnames = list(rows[0]) if rows else [
        "lamp_observation_id",
        "lamp_track_id",
        "clip_id",
        "status",
        "overall_category",
        "overall_score",
        "confidence",
        "attribution_class",
        "physical_valid",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_geojson(path: str | Path, reports: list[MeasurementReport]) -> None:
    features = []
    for report in reports:
        lat = report.geo_summary.get("lat")
        lon = report.geo_summary.get("lon")
        if lat is None or lon is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "lamp_observation_id": report.lamp_observation_id,
                "lamp_track_id": report.lamp_track_id,
                "overall_category": report.metrics["overall_category"],
                "overall_score": report.metrics["overall_useful_illumination_score"],
                "confidence": report.confidence["overall"],
            },
        })
    payload = {"type": "FeatureCollection", "features": features}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
