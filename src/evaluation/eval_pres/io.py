from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import BoxRecord, EvaluationInputs, ReportRecord


def load_json(path: str | Path | None) -> Any:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_manifest_predictions(path: str | Path) -> tuple[BoxRecord, ...]:
    payload = load_json(path)
    rows: list[BoxRecord] = []
    for item in payload.get("tracks", []):
        bbox = _bbox(item.get("bbox_xyxy"))
        if bbox is None:
            continue
        rows.append(
            BoxRecord(
                frame_id=int(item.get("frame_id", 0)),
                bbox_xyxy=bbox,
                score=float(item.get("detector_score", 0.0) or 0.0),
                track_id=str(item.get("track_id")) if item.get("track_id") is not None else None,
                physical_lamp_id=_optional_str(item.get("physical_lamp_id")),
                inventory_id=_optional_str(item.get("inventory_id")),
            )
        )
    return tuple(rows)


def load_reports(path: str | Path | None) -> tuple[ReportRecord, ...]:
    if path is None:
        return ()
    payload = load_json(path)
    if isinstance(payload, dict) and "reports" in payload:
        payload = payload["reports"]
    reports: list[ReportRecord] = []
    for item in payload or []:
        metrics = item.get("metrics", {}) or {}
        status = item.get("status", {}) or {}
        confidence = item.get("confidence", {}) or {}
        affected = item.get("affected_region", {}) or {}
        trace = item.get("traceability", {}) or {}
        reports.append(
            ReportRecord(
                lamp_observation_id=str(item.get("lamp_observation_id") or item.get("id") or ""),
                lamp_track_id=str(item.get("lamp_track_id") or item.get("track_id") or ""),
                mapped_lamp_id=_optional_str(item.get("mapped_lamp_id") or trace.get("mapped_lamp_id")),
                clip_id=str(item.get("clip_id") or ""),
                status_label=_optional_str(status.get("label") or status.get("status_label")),
                overall_category=_optional_str(metrics.get("overall_category")),
                overall_score=_float_or_none(metrics.get("overall_useful_illumination_score")),
                confidence=_float_or_none(confidence.get("overall")),
                affected_region_area_fraction=_float_or_none(
                    affected.get("area_fraction") or affected.get("served_area_fraction") or affected.get("support_fraction")
                ),
                affected_region_polygon=_polygon(affected.get("polygon") or affected.get("points")),
                attribution_class=_optional_str(
                    metrics.get("attribution_class")
                    or item.get("attribution", {}).get("attribution_class")
                    or trace.get("attribution_class")
                ),
                target_attribution_score=_float_or_none(
                    metrics.get("attribution_score")
                    or metrics.get("target_attribution_score")
                    or item.get("attribution", {}).get("score")
                ),
                raw=item,
            )
        )
    return tuple(reports)


def load_ground_truth(path: str | Path | None) -> tuple[BoxRecord, ...]:
    if path is None:
        return ()
    payload = load_json(path)
    rows: list[BoxRecord] = []
    for item in _iter_gt_items(payload):
        bbox = _bbox(item.get("bbox_xyxy") or item.get("bbox") or item.get("box"))
        if bbox is None:
            continue
        rows.append(
            BoxRecord(
                frame_id=int(item.get("frame_id", 0)),
                bbox_xyxy=bbox,
                score=1.0,
                track_id=_optional_str(item.get("track_id")),
                physical_lamp_id=_optional_str(item.get("physical_lamp_id") or item.get("lamp_id") or item.get("id")),
                inventory_id=_optional_str(item.get("inventory_id") or item.get("mapped_lamp_id")),
                status=_optional_str(item.get("status") or item.get("lamp_status")),
                illumination_class=_optional_str(item.get("illumination_class") or item.get("overall_category")),
                affected_region_polygon=_polygon(
                    item.get("affected_region_polygon")
                    or item.get("affected_polygon")
                    or item.get("affected_region", {}).get("polygon")
                    or item.get("affected_region", {}).get("points")
                ),
                served_area_fraction=_float_or_none(
                    item.get("served_area_fraction")
                    or item.get("affected_region_area_fraction")
                    or item.get("coverage_fraction")
                ),
                confounder_present=_bool_or_none(item.get("confounder_present") or item.get("has_confounder")),
                target_attribution_correct=_bool_or_none(
                    item.get("target_attribution_correct") or item.get("target_lamp_attribution_correct")
                ),
                mapped_lamp_id=_optional_str(item.get("mapped_lamp_id")),
            )
        )
    return tuple(rows)


def build_inputs(
    manifest: str | Path,
    reports: str | Path | None,
    ground_truth: str | Path | None,
    route_distance_km: float | None,
    latency_seconds: float | None,
    model_paths: list[str | Path],
) -> EvaluationInputs:
    gt_payload = load_json(ground_truth) if ground_truth else {}
    metadata = gt_payload.get("metadata", {}) if isinstance(gt_payload, dict) else {}
    resolved_route_distance = route_distance_km
    if resolved_route_distance is None:
        resolved_route_distance = _float_or_none(metadata.get("route_distance_km"))
    size_mb = _model_size_mb(model_paths)
    return EvaluationInputs(
        predictions=load_manifest_predictions(manifest),
        ground_truth=load_ground_truth(ground_truth),
        reports=load_reports(reports),
        route_distance_km=resolved_route_distance,
        latency_seconds=latency_seconds,
        model_size_mb=size_mb,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _iter_gt_items(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("lamps", "annotations", "objects", "tracks"):
        for item in payload.get(key, []) or []:
            if isinstance(item, dict):
                rows.append(item)
    for frame in payload.get("frames", []) or []:
        if not isinstance(frame, dict):
            continue
        frame_id = frame.get("frame_id")
        for lamp in frame.get("lamps", []) or frame.get("objects", []) or frame.get("annotations", []) or []:
            if isinstance(lamp, dict):
                merged = dict(lamp)
                merged.setdefault("frame_id", frame_id)
                rows.append(merged)
    return rows


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in value)
    return (x1, y1, x2, y2)


def _polygon(value: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    points: list[tuple[float, float]] = []
    for point in value:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return tuple(points) if len(points) >= 3 else None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _model_size_mb(paths: list[str | Path]) -> float | None:
    total = 0
    found = False
    for raw in paths:
        path = Path(raw)
        if path.exists() and path.is_file():
            total += path.stat().st_size
            found = True
        elif path.exists() and path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    total += child.stat().st_size
                    found = True
    return round(total / (1024 * 1024), 4) if found else None

