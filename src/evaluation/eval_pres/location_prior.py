"""Location-memory prior for streetlight audit runs.

The measurement block aggregates repeated observations into GPS candidates.
This module provides the same idea for the detection audit pipeline: keep an
auditable catalog of places where streetlights have repeatedly been measured,
then query or update that catalog independently from the detector.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.eval_pres.aggregator import AggregatedLamp


SCHEMA_VERSION = "streetlight_location_prior_v1"
REPORT_SCHEMA_VERSION = "streetlight_location_prior_report_v1"
IMPLEMENTATION = "deterministic_location_prior_v1"


@dataclass(frozen=True)
class LocationPriorSettings:
    match_radius_m: float = 12.0
    good_gps_match_radius_m: float = 8.0
    min_observations_for_existing: int = 2
    min_devices_for_high_confidence: int = 2
    existence_confidence_threshold: float = 0.72


@dataclass(frozen=True)
class LocationSample:
    latitude: float
    longitude: float
    frame_index: int | None = None
    time_sec: float | None = None
    gps_accuracy_m: float | None = None
    device_id: str | None = None
    route_group: str | None = None
    source: str = "telemetry"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "time_sec": self.time_sec,
            "latitude": round(self.latitude, 8),
            "longitude": round(self.longitude, 8),
            "gps_accuracy_m": None if self.gps_accuracy_m is None else round(self.gps_accuracy_m, 3),
            "device_id": self.device_id,
            "route_group": self.route_group,
            "source": self.source,
        }


@dataclass(frozen=True)
class LampLocationEvidence:
    observation_id: str
    lamp_track_id: str
    latitude: float
    longitude: float
    gps_accuracy_m: float | None
    detector_confidence: float
    frame_count: int
    first_frame: int
    last_frame: int
    status: str
    device_id: str | None = None
    route_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "lamp_track_id": self.lamp_track_id,
            "latitude": round(self.latitude, 8),
            "longitude": round(self.longitude, 8),
            "gps_accuracy_m": None if self.gps_accuracy_m is None else round(self.gps_accuracy_m, 3),
            "detector_confidence": round(self.detector_confidence, 4),
            "frame_count": self.frame_count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "status": self.status,
            "device_id": self.device_id,
            "route_group": self.route_group,
        }


@dataclass(frozen=True)
class PriorMatch:
    candidate_lamp_id: str
    distance_m: float
    confidence: float
    claim: str
    observation_count: int
    device_count: int
    route_groups: tuple[str, ...]
    latitude: float
    longitude: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_lamp_id": self.candidate_lamp_id,
            "distance_m": round(self.distance_m, 3),
            "confidence": round(self.confidence, 4),
            "claim": self.claim,
            "observation_count": self.observation_count,
            "device_count": self.device_count,
            "route_groups": list(self.route_groups),
            "latitude": round(self.latitude, 8),
            "longitude": round(self.longitude, 8),
        }


@dataclass
class KnownLampCandidate:
    candidate_lamp_id: str
    latitude: float
    longitude: float
    gps_accuracy_m: float | None = None
    confidence: float = 0.0
    observation_count: int = 0
    devices: set[str] = field(default_factory=set)
    route_groups: set[str] = field(default_factory=set)
    status_counts: dict[str, int] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnownLampCandidate":
        geo = payload.get("geo_summary") if isinstance(payload.get("geo_summary"), dict) else payload
        lat = _as_float(geo.get("lat", geo.get("latitude")))
        lon = _as_float(geo.get("lon", geo.get("longitude")))
        if lat is None or lon is None:
            raise ValueError("known lamp candidate requires latitude/longitude")
        confidence = _as_float(payload.get("confidence"))
        if confidence is None:
            consensus = payload.get("consensus_metrics") if isinstance(payload.get("consensus_metrics"), dict) else {}
            confidence = _as_float(consensus.get("overall_useful_illumination_score")) or 0.0
        observations = payload.get("observations", [])
        contributing = payload.get("contributing_observations", [])
        observation_count = _as_int(payload.get("observation_count"))
        if observation_count is None:
            observation_count = len(observations) or len(contributing) or 1
        devices = _as_str_set(payload.get("devices") or payload.get("device_ids"))
        routes = _as_str_set(payload.get("route_groups"))
        return cls(
            candidate_lamp_id=str(
                payload.get("candidate_lamp_id")
                or payload.get("lamp_id")
                or payload.get("id")
                or "candidate:gps:imported"
            ),
            latitude=lat,
            longitude=lon,
            gps_accuracy_m=_as_float(geo.get("gps_accuracy_m")),
            confidence=max(0.0, min(1.0, confidence)),
            observation_count=max(0, observation_count),
            devices=devices,
            route_groups=routes,
            status_counts={str(k): int(v) for k, v in payload.get("status_counts", {}).items()},
            observations=list(observations) if isinstance(observations, list) else [],
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            provenance=dict(payload.get("provenance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_lamp_id": self.candidate_lamp_id,
            "geo_summary": {
                "lat": round(self.latitude, 8),
                "lon": round(self.longitude, 8),
                "gps_accuracy_m": None if self.gps_accuracy_m is None else round(self.gps_accuracy_m, 3),
            },
            "confidence": round(self.confidence, 4),
            "observation_count": self.observation_count,
            "devices": sorted(self.devices),
            "route_groups": sorted(self.route_groups),
            "status_counts": dict(sorted(self.status_counts.items())),
            "observations": self.observations[-20:],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance": self.provenance,
        }

    def merge_observation(self, evidence: LampLocationEvidence, settings: LocationPriorSettings) -> None:
        now = datetime.now().isoformat()
        previous_weight = max(1.0, float(self.observation_count))
        evidence_weight = max(0.05, min(1.0, evidence.detector_confidence))
        total_weight = previous_weight + evidence_weight
        self.latitude = ((self.latitude * previous_weight) + (evidence.latitude * evidence_weight)) / total_weight
        self.longitude = ((self.longitude * previous_weight) + (evidence.longitude * evidence_weight)) / total_weight
        if evidence.gps_accuracy_m is not None:
            if self.gps_accuracy_m is None:
                self.gps_accuracy_m = evidence.gps_accuracy_m
            else:
                self.gps_accuracy_m = min(self.gps_accuracy_m, evidence.gps_accuracy_m)
        self.observation_count += 1
        if evidence.device_id:
            self.devices.add(evidence.device_id)
        if evidence.route_group:
            self.route_groups.add(evidence.route_group)
        self.status_counts[evidence.status] = self.status_counts.get(evidence.status, 0) + 1
        self.observations.append(evidence.to_dict())
        self.observations = self.observations[-20:]
        if self.created_at is None:
            self.created_at = now
        self.updated_at = now
        self.confidence = _existence_confidence(
            detector_confidence=evidence.detector_confidence,
            observation_count=self.observation_count,
            device_count=len(self.devices),
            gps_accuracy_m=self.gps_accuracy_m,
            previous_confidence=self.confidence,
            settings=settings,
        )
        self.provenance = {"implementation": IMPLEMENTATION, "aggregation": "gps_radius_device_aware_consensus"}


class LocationPriorStore:
    def __init__(self, candidates: list[KnownLampCandidate] | None = None) -> None:
        self.candidates = candidates or []

    @classmethod
    def load(cls, path: str | Path | None) -> "LocationPriorStore":
        if not path:
            return cls()
        prior_path = Path(path)
        if not prior_path.exists():
            return cls()
        payload = json.loads(prior_path.read_text(encoding="utf-8-sig"))
        return cls(_parse_candidates(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "implementation": IMPLEMENTATION,
            "known_lamps": [candidate.to_dict() for candidate in sorted(self.candidates, key=lambda item: item.candidate_lamp_id)],
        }

    def write(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def best_match(
        self,
        latitude: float,
        longitude: float,
        gps_accuracy_m: float | None,
        settings: LocationPriorSettings,
    ) -> PriorMatch | None:
        best: tuple[KnownLampCandidate, float] | None = None
        for candidate in self.candidates:
            radius = _match_radius(gps_accuracy_m, candidate.gps_accuracy_m, settings)
            distance = haversine_m(latitude, longitude, candidate.latitude, candidate.longitude)
            if distance <= radius and (best is None or distance < best[1]):
                best = (candidate, distance)
        if best is None:
            return None
        candidate, distance = best
        radius = _match_radius(gps_accuracy_m, candidate.gps_accuracy_m, settings)
        distance_penalty = min(0.35, 0.35 * (distance / max(radius, 0.001)))
        confidence = max(0.0, min(1.0, candidate.confidence * (1.0 - distance_penalty)))
        return PriorMatch(
            candidate_lamp_id=candidate.candidate_lamp_id,
            distance_m=distance,
            confidence=confidence,
            claim=_claim_label(confidence, candidate.observation_count, len(candidate.devices), settings),
            observation_count=candidate.observation_count,
            device_count=len(candidate.devices),
            route_groups=tuple(sorted(candidate.route_groups)),
            latitude=candidate.latitude,
            longitude=candidate.longitude,
        )

    def update_with_observation(
        self,
        evidence: LampLocationEvidence,
        settings: LocationPriorSettings,
    ) -> tuple[KnownLampCandidate, PriorMatch | None, bool]:
        match = self.best_match(evidence.latitude, evidence.longitude, evidence.gps_accuracy_m, settings)
        if match is not None:
            candidate = self._candidate_by_id(match.candidate_lamp_id)
            candidate.merge_observation(evidence, settings)
            updated_match = self.best_match(evidence.latitude, evidence.longitude, evidence.gps_accuracy_m, settings)
            return candidate, updated_match, False

        candidate = KnownLampCandidate(
            candidate_lamp_id=self._next_candidate_id(),
            latitude=evidence.latitude,
            longitude=evidence.longitude,
            gps_accuracy_m=evidence.gps_accuracy_m,
        )
        candidate.merge_observation(evidence, settings)
        self.candidates.append(candidate)
        return candidate, None, True

    def _candidate_by_id(self, candidate_lamp_id: str) -> KnownLampCandidate:
        for candidate in self.candidates:
            if candidate.candidate_lamp_id == candidate_lamp_id:
                return candidate
        raise KeyError(candidate_lamp_id)

    def _next_candidate_id(self) -> str:
        existing = {candidate.candidate_lamp_id for candidate in self.candidates}
        index = 1
        while True:
            candidate_id = f"candidate:gps:{index:04d}"
            if candidate_id not in existing:
                return candidate_id
            index += 1


def load_location_samples(path: str | Path | None) -> list[LocationSample]:
    if not path:
        return []
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Location samples not found: {sample_path}")
    if sample_path.suffix.lower() == ".csv":
        with sample_path.open("r", newline="", encoding="utf-8-sig") as handle:
            return [_sample_from_row(row, source=str(sample_path)) for row in csv.DictReader(handle)]

    payload = json.loads(sample_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        rows = payload
        defaults: dict[str, Any] = {}
    else:
        rows = payload.get("samples") or payload.get("frames") or []
        defaults = {
            "device_id": payload.get("device_id"),
            "route_group": _route_group_from_payload(payload),
        }
    return [_sample_from_row(row, defaults=defaults, source=str(sample_path)) for row in rows]


def static_location_sample(
    latitude: float | None,
    longitude: float | None,
    gps_accuracy_m: float | None,
    device_id: str | None,
    route_group: str | None,
) -> list[LocationSample]:
    if latitude is None or longitude is None:
        return []
    return [
        LocationSample(
            latitude=float(latitude),
            longitude=float(longitude),
            gps_accuracy_m=gps_accuracy_m,
            device_id=device_id,
            route_group=route_group,
            source="static_capture_location",
        )
    ]


def evidence_for_lamp(
    lamp: AggregatedLamp,
    samples: list[LocationSample],
    *,
    run_id: str,
    default_device_id: str | None = None,
    default_route_group: str | None = None,
) -> LampLocationEvidence | None:
    if not samples:
        return None
    in_window = [
        sample
        for sample in samples
        if sample.frame_index is None or lamp.first_frame <= sample.frame_index <= lamp.last_frame
    ]
    if not in_window:
        return None
    latitudes = [sample.latitude for sample in in_window]
    longitudes = [sample.longitude for sample in in_window]
    accuracies = [sample.gps_accuracy_m for sample in in_window if sample.gps_accuracy_m is not None]
    device_id = next((sample.device_id for sample in in_window if sample.device_id), default_device_id)
    route_group = next((sample.route_group for sample in in_window if sample.route_group), default_route_group)
    return LampLocationEvidence(
        observation_id=f"obs_{run_id}_{lamp.track_id}",
        lamp_track_id=lamp.track_id,
        latitude=sum(latitudes) / len(latitudes),
        longitude=sum(longitudes) / len(longitudes),
        gps_accuracy_m=sum(accuracies) / len(accuracies) if accuracies else None,
        detector_confidence=lamp.confidence,
        frame_count=lamp.frame_count,
        first_frame=lamp.first_frame,
        last_frame=lamp.last_frame,
        status=lamp.status,
        device_id=device_id,
        route_group=route_group,
    )


def build_location_prior_report(
    *,
    prior_path: str | None,
    updated_prior_path: str | None,
    query: dict[str, Any] | None,
    query_match: PriorMatch | None,
    lamp_updates: list[dict[str, Any]],
    store: LocationPriorStore,
    settings: LocationPriorSettings,
) -> dict[str, Any]:
    matched_existing = sum(1 for item in lamp_updates if not item.get("new_candidate"))
    new_candidates = sum(1 for item in lamp_updates if item.get("new_candidate"))
    known_existing = sum(
        1
        for candidate in store.candidates
        if _claim_label(candidate.confidence, candidate.observation_count, len(candidate.devices), settings)
        == "known_lamp_likely_exists"
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "implementation": IMPLEMENTATION,
        "prior_path": prior_path,
        "updated_prior_path": updated_prior_path,
        "settings": {
            "match_radius_m": settings.match_radius_m,
            "good_gps_match_radius_m": settings.good_gps_match_radius_m,
            "min_observations_for_existing": settings.min_observations_for_existing,
            "min_devices_for_high_confidence": settings.min_devices_for_high_confidence,
            "existence_confidence_threshold": settings.existence_confidence_threshold,
        },
        "query": {
            **(query or {}),
            "match": query_match.to_dict() if query_match else None,
            "claim": query_match.claim if query_match else "no_known_lamp_at_location",
        },
        "current_run_updates": {
            "location_evidence_count": len(lamp_updates),
            "matched_existing_candidates": matched_existing,
            "new_candidates": new_candidates,
            "lamp_updates": lamp_updates,
        },
        "prior_summary": {
            "known_lamp_candidates": len(store.candidates),
            "known_lamp_likely_exists": known_existing,
        },
    }
    return report


def write_location_prior_report(output_dir: str | Path, report: dict[str, Any]) -> Path:
    path = Path(output_dir) / "location_prior_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_candidates(payload: Any) -> list[KnownLampCandidate]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("known_lamps") or payload.get("lamps") or payload.get("candidates") or []
    else:
        rows = []
    candidates: list[KnownLampCandidate] = []
    for row in rows:
        try:
            candidates.append(KnownLampCandidate.from_dict(row))
        except ValueError:
            continue
    return candidates


def _sample_from_row(
    row: dict[str, Any],
    defaults: dict[str, Any] | None = None,
    source: str = "telemetry",
) -> LocationSample:
    defaults = defaults or {}
    pose = row.get("pose") if isinstance(row.get("pose"), dict) else {}
    lat = _as_float(row.get("latitude", row.get("lat", pose.get("latitude"))))
    lon = _as_float(row.get("longitude", row.get("lon", pose.get("longitude"))))
    if lat is None or lon is None:
        raise ValueError("location sample requires latitude/longitude")
    return LocationSample(
        latitude=lat,
        longitude=lon,
        frame_index=_as_int(row.get("frame_index", row.get("frame_id"))),
        time_sec=_as_float(row.get("time_sec", row.get("timestamp_sec"))),
        gps_accuracy_m=_as_float(row.get("gps_accuracy_m", row.get("accuracy_m", pose.get("gps_accuracy_m")))),
        device_id=row.get("device_id") or defaults.get("device_id"),
        route_group=row.get("route_group") or defaults.get("route_group"),
        source=source,
    )


def _route_group_from_payload(payload: dict[str, Any]) -> str | None:
    if payload.get("route_group"):
        return str(payload["route_group"])
    calibration = payload.get("optional_calibration")
    if not isinstance(calibration, dict):
        return None
    map_priors = calibration.get("map_priors")
    if not isinstance(map_priors, dict):
        return None
    value = map_priors.get("route_group")
    return str(value) if value else None


def _match_radius(
    query_accuracy_m: float | None,
    candidate_accuracy_m: float | None,
    settings: LocationPriorSettings,
) -> float:
    if _good_gps(query_accuracy_m) and _good_gps(candidate_accuracy_m):
        return settings.good_gps_match_radius_m
    accuracy_pad = max(query_accuracy_m or 0.0, candidate_accuracy_m or 0.0) * 0.25
    return max(settings.match_radius_m, settings.good_gps_match_radius_m + accuracy_pad)


def _existence_confidence(
    *,
    detector_confidence: float,
    observation_count: int,
    device_count: int,
    gps_accuracy_m: float | None,
    previous_confidence: float,
    settings: LocationPriorSettings,
) -> float:
    observation_support = min(1.0, observation_count / max(1, settings.min_observations_for_existing))
    device_support = min(1.0, device_count / max(1, settings.min_devices_for_high_confidence))
    gps_support = _gps_quality(gps_accuracy_m)
    detector_support = max(0.0, min(1.0, detector_confidence))
    new_confidence = (
        0.42 * detector_support
        + 0.28 * observation_support
        + 0.20 * device_support
        + 0.10 * gps_support
    )
    if previous_confidence > 0:
        new_confidence = max(new_confidence, (0.65 * previous_confidence) + (0.35 * new_confidence))
    return round(max(0.0, min(1.0, new_confidence)), 4)


def _claim_label(
    confidence: float,
    observation_count: int,
    device_count: int,
    settings: LocationPriorSettings,
) -> str:
    enough_observations = observation_count >= settings.min_observations_for_existing
    enough_devices = device_count >= settings.min_devices_for_high_confidence
    if confidence >= settings.existence_confidence_threshold and (enough_observations or enough_devices):
        return "known_lamp_likely_exists"
    if confidence >= 0.5:
        return "probable_lamp_location"
    return "weak_location_prior"


def _gps_quality(accuracy_m: float | None) -> float:
    if accuracy_m is None:
        return 0.45
    if accuracy_m <= 5:
        return 1.0
    if accuracy_m >= 30:
        return 0.2
    return max(0.2, 1.0 - ((accuracy_m - 5.0) / 25.0) * 0.8)


def _good_gps(accuracy_m: float | None) -> bool:
    return accuracy_m is not None and accuracy_m <= 5.0


def _as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item not in {None, ""}}
    return set()
