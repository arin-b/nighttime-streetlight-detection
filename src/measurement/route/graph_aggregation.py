from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.module_io import AggregatedLampReport, AuditTrailArtifact, RoadSegmentReport, RouteGraphArtifact
from rbccps_measurement.contracts.output_schema import MeasurementReport


IMPLEMENTATION = "deterministic_route_graph_aggregation_v1"
ORDINAL_CLASSES = ("unknown", "poor", "marginal", "adequate")
EDGE_TYPES = ("observed_as", "same_candidate_as", "within_segment", "from_pass", "near_gps", "has_map_prior")


@dataclass(frozen=True)
class RouteAggregationConfig:
    gps_merge_radius_m: float = 12.0
    good_gps_merge_radius_m: float = 8.0
    disagreement_review_threshold: float = 0.42
    underlighting_threshold: float = 0.45
    geo_bin_degrees: float = 0.001

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class RouteAggregationOutput:
    graph: RouteGraphArtifact
    lamps: tuple[AggregatedLampReport, ...]
    road_segments: tuple[RoadSegmentReport, ...]
    audit_trail: AuditTrailArtifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": asdict(self.graph),
            "lamps": [asdict(item) for item in self.lamps],
            "road_segments": [asdict(item) for item in self.road_segments],
            "audit_trail": asdict(self.audit_trail),
        }


def aggregate_by_lamp_track(reports: list[MeasurementReport]) -> dict[str, list[MeasurementReport]]:
    grouped: dict[str, list[MeasurementReport]] = defaultdict(list)
    for report in reports:
        grouped[report.lamp_track_id].append(report)
    return dict(grouped)


def aggregate_route_reports(
    reports: list[MeasurementReport],
    source_report_refs: dict[str, str] | None = None,
    route_group: str | None = None,
    config: RouteAggregationConfig | None = None,
) -> RouteAggregationOutput:
    config = config or RouteAggregationConfig()
    source_report_refs = source_report_refs or {}
    unique_reports = _dedupe_observations(reports)
    assignments = _assign_candidates(unique_reports, config)
    segments = {report.lamp_observation_id: _segment_id(report, route_group, config) for report in unique_reports}

    lamp_reports = tuple(
        _aggregate_candidate(candidate_id, candidate_reports, segments, config)
        for candidate_id, candidate_reports in sorted(assignments.items())
    )
    segment_reports = tuple(_aggregate_segments(lamp_reports, segments, route_group))
    graph = _build_graph(unique_reports, assignments, segments, lamp_reports, segment_reports, route_group, config)
    audit = build_audit_trail(unique_reports, source_report_refs, route_group)
    return RouteAggregationOutput(graph=graph, lamps=lamp_reports, road_segments=segment_reports, audit_trail=audit)


def build_audit_trail(
    reports: list[MeasurementReport],
    source_report_refs: dict[str, str] | None = None,
    route_group: str | None = None,
) -> AuditTrailArtifact:
    source_report_refs = source_report_refs or {}
    module_versions: dict[str, Any] = {}
    flags: set[str] = set()
    evidence_refs: list[dict[str, Any]] = []
    physical_valid = 0
    physical_invalid = 0
    for report in reports:
        module_versions.update(report.traceability.get("model_versions", {}))
        flags.update(report.uncertainty_flags)
        flags.update(report.optional_physical_estimates.get("quality_flags", []))
        if report.optional_physical_estimates.get("valid"):
            physical_valid += 1
        else:
            physical_invalid += 1
        evidence_refs.append(
            {
                "lamp_observation_id": report.lamp_observation_id,
                "clip_id": report.clip_id,
                "mask_ref": report.affected_region.get("image_mask_uri"),
                "feature_snapshot_ref": report.traceability.get("feature_snapshot_ref"),
                "source_report_ref": source_report_refs.get(report.lamp_observation_id),
                "policy_id": report.traceability.get("policy_id"),
            }
        )
    return AuditTrailArtifact(
        source_reports=tuple(
            {
                "lamp_observation_id": report.lamp_observation_id,
                "clip_id": report.clip_id,
                "source_report_ref": source_report_refs.get(report.lamp_observation_id),
            }
            for report in reports
        ),
        module_versions=module_versions,
        evidence_refs=tuple(evidence_refs),
        quality_flags=tuple(sorted(flags)),
        calibration_summary={
            "physical_valid_reports": physical_valid,
            "physical_invalid_reports": physical_invalid,
            "calibration_levels": sorted({report.confidence.get("calibration_level") for report in reports}),
        },
        generation_metadata={
            "implementation": IMPLEMENTATION,
            "route_group": route_group or "unknown_route",
            "report_count": len(reports),
            "audit_payload": "compact_references_not_full_intermediate_arrays",
        },
    )


def write_route_aggregation_outputs(output: RouteAggregationOutput, out: str | Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "route_aggregation.json").write_text(json.dumps(output.to_dict(), indent=2), encoding="utf-8")
    (out / "audit_trail.json").write_text(json.dumps(asdict(output.audit_trail), indent=2), encoding="utf-8")
    _write_lamps_geojson(out / "route_lamps.geojson", output.lamps)
    _write_segments_geojson(out / "road_segments.geojson", output.road_segments)


def _dedupe_observations(reports: list[MeasurementReport]) -> list[MeasurementReport]:
    by_id: dict[str, MeasurementReport] = {}
    for report in reports:
        existing = by_id.get(report.lamp_observation_id)
        if existing is None or float(report.confidence.get("overall", 0.0)) > float(existing.confidence.get("overall", 0.0)):
            by_id[report.lamp_observation_id] = report
    return list(by_id.values())


def _assign_candidates(reports: list[MeasurementReport], config: RouteAggregationConfig) -> dict[str, list[MeasurementReport]]:
    assignments: dict[str, list[MeasurementReport]] = defaultdict(list)
    gps_candidates: list[tuple[str, float, float, float | None]] = []
    for report in reports:
        mapped = report.mapped_lamp_id or report.traceability.get("mapped_lamp_id") or report.traceability.get("lamp_inventory_id")
        lat = _as_float(report.geo_summary.get("lat"))
        lon = _as_float(report.geo_summary.get("lon"))
        accuracy = _as_float(report.geo_summary.get("gps_accuracy_m"))
        if mapped:
            candidate_id = f"inventory:{mapped}"
        elif lat is not None and lon is not None:
            candidate_id = _nearest_gps_candidate(lat, lon, accuracy, gps_candidates, config)
            if candidate_id is None:
                candidate_id = f"candidate:gps:{len(gps_candidates) + 1:04d}"
                gps_candidates.append((candidate_id, lat, lon, accuracy))
        else:
            candidate_id = f"candidate:observation:{report.lamp_observation_id}"
        assignments[candidate_id].append(report)
    return dict(assignments)


def _nearest_gps_candidate(
    lat: float,
    lon: float,
    accuracy: float | None,
    candidates: list[tuple[str, float, float, float | None]],
    config: RouteAggregationConfig,
) -> str | None:
    best: tuple[str, float] | None = None
    for candidate_id, candidate_lat, candidate_lon, candidate_accuracy in candidates:
        radius = config.good_gps_merge_radius_m if _good_gps(accuracy) and _good_gps(candidate_accuracy) else config.gps_merge_radius_m
        distance = _haversine_m(lat, lon, candidate_lat, candidate_lon)
        if distance <= radius and (best is None or distance < best[1]):
            best = (candidate_id, distance)
    return best[0] if best else None


def _aggregate_candidate(
    candidate_id: str,
    reports: list[MeasurementReport],
    segments: dict[str, str],
    config: RouteAggregationConfig,
) -> AggregatedLampReport:
    weights = [max(0.05, float(report.confidence.get("overall", 0.0))) for report in reports]
    scores = [float(report.metrics.get("overall_useful_illumination_score", 0.0)) for report in reports]
    weighted_score = _weighted_mean(scores, weights)
    category = _category(weighted_score)
    histogram = _category_histogram(reports, weights)
    disagreement = _disagreement_score(reports, weighted_score)
    flags = sorted({flag for report in reports for flag in report.uncertainty_flags})
    if disagreement >= config.disagreement_review_threshold:
        flags.append("route_category_disagreement")
    if any(report.confidence.get("action") == "abstain" for report in reports):
        flags.append("contains_abstained_observation")
    priority = _manual_review_priority(disagreement, weighted_score, flags)
    physical = _aggregate_physical(reports)
    return AggregatedLampReport(
        candidate_lamp_id=candidate_id,
        contributing_observations=tuple(report.lamp_observation_id for report in reports),
        contributing_clips=tuple(sorted({report.clip_id for report in reports})),
        geo_summary=_geo_summary(reports),
        consensus_metrics={
            "overall_useful_illumination_score": round(weighted_score, 4),
            "overall_category": category,
            "confidence_weighted_observations": round(sum(weights), 4),
            "worst_credible_category": _worst_category(reports),
            "road_segment_ids": sorted({segments[report.lamp_observation_id] for report in reports}),
        },
        category_histogram={key: round(value, 4) for key, value in histogram.items()},
        physical_estimate_summary=physical,
        disagreement_score=round(disagreement, 4),
        manual_review_priority=priority,
        quality_flags=tuple(sorted(set(flags))),
        provenance={"implementation": IMPLEMENTATION, "aggregation": "confidence_weighted_consensus_with_outlier_flags"},
    )


def _aggregate_segments(
    lamps: tuple[AggregatedLampReport, ...],
    observation_segments: dict[str, str],
    route_group: str | None,
) -> list[RoadSegmentReport]:
    by_segment: dict[str, list[AggregatedLampReport]] = defaultdict(list)
    for lamp in lamps:
        segment_ids = lamp.consensus_metrics.get("road_segment_ids") or ["segment:unknown"]
        for segment_id in segment_ids:
            by_segment[str(segment_id)].append(lamp)
    reports: list[RoadSegmentReport] = []
    for segment_id, segment_lamps in sorted(by_segment.items()):
        scores = [float(lamp.consensus_metrics["overall_useful_illumination_score"]) for lamp in segment_lamps]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        worst = min((_category_rank(str(lamp.consensus_metrics["overall_category"])), str(lamp.consensus_metrics["overall_category"])) for lamp in segment_lamps)[1]
        underlighting = sum(1.0 - score for score in scores) / len(scores) if scores else 0.0
        flags = ["synthetic_segment"] if segment_id.startswith("segment:synthetic") else []
        if underlighting >= 0.55 or worst in {"unknown", "poor"}:
            flags.append("underlighting_review_recommended")
        reports.append(
            RoadSegmentReport(
                segment_id=segment_id,
                route_group=route_group or "unknown_route",
                observation_count=sum(len(lamp.contributing_observations) for lamp in segment_lamps),
                candidate_lamp_count=len(segment_lamps),
                mean_score=round(mean_score, 4),
                worst_category=worst,
                underlighting_score=round(underlighting, 4),
                manual_review_priority="high" if "underlighting_review_recommended" in flags else "normal",
                quality_flags=tuple(flags),
                provenance={"implementation": IMPLEMENTATION},
            )
        )
    return reports


def _build_graph(
    reports: list[MeasurementReport],
    assignments: dict[str, list[MeasurementReport]],
    segments: dict[str, str],
    lamps: tuple[AggregatedLampReport, ...],
    road_segments: tuple[RoadSegmentReport, ...],
    route_group: str | None,
    config: RouteAggregationConfig,
) -> RouteGraphArtifact:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for report in reports:
        obs_id = f"observation:{report.lamp_observation_id}"
        pass_id = f"pass:{report.clip_id}"
        gps_id = f"gps:{_gps_bin(report, config)}"
        segment_id = segments[report.lamp_observation_id]
        nodes.extend(
            [
                {"id": obs_id, "type": "observation", "features": _observation_features(report)},
                {"id": pass_id, "type": "drive_pass", "features": {"clip_id": report.clip_id}},
                {"id": gps_id, "type": "gps_neighborhood", "features": {"bin": gps_id}},
            ]
        )
        edges.extend(
            [
                {"source": obs_id, "target": pass_id, "type": "from_pass", "weight": 1.0},
                {"source": obs_id, "target": gps_id, "type": "near_gps", "weight": 1.0},
                {"source": obs_id, "target": segment_id, "type": "within_segment", "weight": 1.0},
            ]
        )
    for lamp in lamps:
        candidate_node = {"id": lamp.candidate_lamp_id, "type": "candidate_lamp", "features": lamp.consensus_metrics}
        nodes.append(candidate_node)
        if lamp.candidate_lamp_id.startswith("inventory:"):
            map_prior_id = f"map_prior:{lamp.candidate_lamp_id.removeprefix('inventory:')}"
            nodes.append({"id": map_prior_id, "type": "map_prior", "features": {"source": "mapped_lamp_id"}})
            edges.append({"source": lamp.candidate_lamp_id, "target": map_prior_id, "type": "has_map_prior", "weight": 1.0})
        for observation_id in lamp.contributing_observations:
            obs_node = f"observation:{observation_id}"
            edges.append({"source": obs_node, "target": lamp.candidate_lamp_id, "type": "observed_as", "weight": 1.0})
        if len(lamp.contributing_observations) > 1:
            first = f"observation:{lamp.contributing_observations[0]}"
            for other in lamp.contributing_observations[1:]:
                edges.append({"source": first, "target": f"observation:{other}", "type": "same_candidate_as", "weight": 1.0})
    for segment in road_segments:
        nodes.append({"id": segment.segment_id, "type": "road_segment", "features": asdict(segment)})
    nodes = _dedupe_nodes(nodes)
    return RouteGraphArtifact(
        nodes=tuple(nodes),
        edges=tuple(edges),
        graph_statistics={
            "node_count": float(len(nodes)),
            "edge_count": float(len(edges)),
            "observation_count": float(len(reports)),
            "candidate_lamp_count": float(len(lamps)),
            "road_segment_count": float(len(road_segments)),
        },
        route_metadata={"route_group": route_group or "unknown_route", "edge_types": list(EDGE_TYPES), "config": config.to_dict()},
        provenance={"implementation": IMPLEMENTATION},
    )


def _segment_id(report: MeasurementReport, route_group: str | None, config: RouteAggregationConfig) -> str:
    trace = report.traceability or {}
    segment = trace.get("road_segment_id") or trace.get("segment_id")
    if segment:
        return f"segment:{segment}"
    lat = _as_float(report.geo_summary.get("lat"))
    lon = _as_float(report.geo_summary.get("lon"))
    if lat is not None and lon is not None:
        return f"segment:synthetic:{route_group or 'unknown_route'}:{round(lat / config.geo_bin_degrees)}:{round(lon / config.geo_bin_degrees)}"
    return f"segment:synthetic:{route_group or 'unknown_route'}:unknown"


def _gps_bin(report: MeasurementReport, config: RouteAggregationConfig) -> str:
    lat = _as_float(report.geo_summary.get("lat"))
    lon = _as_float(report.geo_summary.get("lon"))
    if lat is None or lon is None:
        return "unknown"
    return f"{round(lat / config.geo_bin_degrees)}:{round(lon / config.geo_bin_degrees)}"


def _observation_features(report: MeasurementReport) -> dict[str, Any]:
    return {
        "score": float(report.metrics.get("overall_useful_illumination_score", 0.0)),
        "category": report.metrics.get("overall_category", "unknown"),
        "confidence": report.confidence.get("overall", 0.0),
        "physical_valid": bool(report.optional_physical_estimates.get("valid")),
    }


def _aggregate_physical(reports: list[MeasurementReport]) -> dict[str, Any]:
    valid = [report for report in reports if report.optional_physical_estimates.get("valid")]
    if not valid:
        return {"valid": False, "reason": "No contributing observation has valid physical estimates.", "valid_observations": 0}
    values = [float(report.optional_physical_estimates["horizontal_illuminance_lux_mean"]) for report in valid if report.optional_physical_estimates.get("horizontal_illuminance_lux_mean") is not None]
    intervals = [report.optional_physical_estimates.get("horizontal_illuminance_lux_interval") for report in valid]
    intervals = [item for item in intervals if isinstance(item, (list, tuple)) and len(item) == 2]
    return {
        "valid": bool(values),
        "valid_observations": len(valid),
        "horizontal_illuminance_lux_mean": round(sum(values) / len(values), 4) if values else None,
        "horizontal_illuminance_lux_interval": [round(min(item[0] for item in intervals), 4), round(max(item[1] for item in intervals), 4)] if intervals else None,
        "reason": "Aggregated only from individually valid physical estimates.",
    }


def _geo_summary(reports: list[MeasurementReport]) -> dict[str, Any]:
    lats = [_as_float(report.geo_summary.get("lat")) for report in reports]
    lons = [_as_float(report.geo_summary.get("lon")) for report in reports]
    accs = [_as_float(report.geo_summary.get("gps_accuracy_m")) for report in reports]
    return {
        "lat": _mean([value for value in lats if value is not None]),
        "lon": _mean([value for value in lons if value is not None]),
        "gps_accuracy_m": _mean([value for value in accs if value is not None]),
    }


def _category_histogram(reports: list[MeasurementReport], weights: list[float]) -> dict[str, float]:
    counts = {label: 0.0 for label in ORDINAL_CLASSES}
    for report, weight in zip(reports, weights, strict=True):
        category = str(report.metrics.get("overall_category", "unknown"))
        counts[category if category in counts else "unknown"] += weight
    total = sum(counts.values()) or 1.0
    return {label: value / total for label, value in counts.items()}


def _disagreement_score(reports: list[MeasurementReport], weighted_score: float) -> float:
    if len(reports) <= 1:
        return 0.0
    values = [float(report.metrics.get("overall_useful_illumination_score", 0.0)) for report in reports]
    score_spread = max(values) - min(values)
    category_count = len({str(report.metrics.get("overall_category", "unknown")) for report in reports})
    category_disagreement = (category_count - 1) / max(1, len(ORDINAL_CLASSES) - 1)
    return max(0.0, min(1.0, 0.65 * score_spread + 0.35 * category_disagreement))


def _manual_review_priority(disagreement: float, score: float, flags: list[str]) -> str:
    if disagreement >= 0.42 or score < 0.25 or "contains_abstained_observation" in flags:
        return "high"
    if disagreement >= 0.35 or score < 0.45 or flags:
        return "medium"
    return "normal"


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def _category(score: float) -> str:
    if score >= 0.72:
        return "adequate"
    if score >= 0.45:
        return "marginal"
    if score >= 0.2:
        return "poor"
    return "unknown"


def _worst_category(reports: list[MeasurementReport]) -> str:
    categories = [str(report.metrics.get("overall_category", "unknown")) for report in reports]
    return min(categories, key=_category_rank) if categories else "unknown"


def _category_rank(category: str) -> int:
    try:
        return ORDINAL_CLASSES.index(category)
    except ValueError:
        return 0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _good_gps(accuracy: float | None) -> bool:
    return accuracy is not None and accuracy <= 5.0


def _as_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 8) if values else None


def _dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        by_id[str(node["id"])] = node
    return list(by_id.values())


def _write_lamps_geojson(path: Path, lamps: tuple[AggregatedLampReport, ...]) -> None:
    features = []
    for lamp in lamps:
        lat = lamp.geo_summary.get("lat")
        lon = lamp.geo_summary.get("lon")
        if lat is None or lon is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "candidate_lamp_id": lamp.candidate_lamp_id,
                    "overall_category": lamp.consensus_metrics["overall_category"],
                    "overall_score": lamp.consensus_metrics["overall_useful_illumination_score"],
                    "manual_review_priority": lamp.manual_review_priority,
                    "observations": len(lamp.contributing_observations),
                },
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def _write_segments_geojson(path: Path, segments: tuple[RoadSegmentReport, ...]) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": None,
            "properties": asdict(segment),
        }
        for segment in segments
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
