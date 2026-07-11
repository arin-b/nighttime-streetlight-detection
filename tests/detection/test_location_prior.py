import json
from pathlib import Path

from audit_pipeline.aggregator import AggregatedLamp
from audit_pipeline.location_prior import (
    LampLocationEvidence,
    LocationPriorSettings,
    LocationPriorStore,
    LocationSample,
    evidence_for_lamp,
)


def make_evidence(
    observation_id: str,
    latitude: float,
    longitude: float,
    *,
    device_id: str = "phone_a",
    confidence: float = 0.9,
) -> LampLocationEvidence:
    return LampLocationEvidence(
        observation_id=observation_id,
        lamp_track_id=f"track_{observation_id}",
        latitude=latitude,
        longitude=longitude,
        gps_accuracy_m=4.0,
        detector_confidence=confidence,
        frame_count=8,
        first_frame=10,
        last_frame=18,
        status="working",
        device_id=device_id,
        route_group="route_a",
    )


def test_location_prior_merges_nearby_multi_device_observations():
    settings = LocationPriorSettings()
    store = LocationPriorStore()

    first_candidate, first_match, first_new = store.update_with_observation(
        make_evidence("a", 12.9716, 77.5946, device_id="phone_a"),
        settings,
    )
    second_candidate, second_match, second_new = store.update_with_observation(
        make_evidence("b", 12.97162, 77.59462, device_id="phone_b"),
        settings,
    )

    assert first_new is True
    assert first_match is None
    assert second_new is False
    assert second_candidate.candidate_lamp_id == first_candidate.candidate_lamp_id
    assert second_match is not None
    assert second_match.claim == "known_lamp_likely_exists"
    assert second_match.device_count == 2

    query = store.best_match(12.97161, 77.59461, 5.0, settings)

    assert query is not None
    assert query.candidate_lamp_id == first_candidate.candidate_lamp_id
    assert query.claim == "known_lamp_likely_exists"


def test_location_prior_keeps_far_observation_as_new_candidate():
    settings = LocationPriorSettings()
    store = LocationPriorStore()

    store.update_with_observation(make_evidence("a", 12.9716, 77.5946), settings)
    far_candidate, _, far_new = store.update_with_observation(
        make_evidence("far", 12.9816, 77.6046),
        settings,
    )

    assert far_new is True
    assert far_candidate.candidate_lamp_id == "candidate:gps:0002"
    assert len(store.candidates) == 2


def test_evidence_for_lamp_uses_samples_inside_track_window():
    lamp = AggregatedLamp(
        track_id="track_1",
        status="working",
        confidence=0.82,
        frame_count=6,
        first_frame=10,
        last_frame=15,
        frames_on=6,
        frames_off=0,
    )
    samples = [
        LocationSample(latitude=10.0, longitude=20.0, frame_index=4, gps_accuracy_m=4.0),
        LocationSample(latitude=12.0, longitude=77.0, frame_index=10, gps_accuracy_m=6.0, device_id="phone_a"),
        LocationSample(latitude=14.0, longitude=79.0, frame_index=15, gps_accuracy_m=4.0, route_group="route_a"),
    ]

    evidence = evidence_for_lamp(lamp, samples, run_id="clip_a", default_device_id="fallback_phone")

    assert evidence is not None
    assert evidence.observation_id == "obs_clip_a_track_1"
    assert evidence.latitude == 13.0
    assert evidence.longitude == 78.0
    assert evidence.gps_accuracy_m == 5.0
    assert evidence.device_id == "phone_a"
    assert evidence.route_group == "route_a"


def test_location_prior_loads_measurement_route_aggregation(tmp_path: Path):
    route_aggregation = {
        "lamps": [
            {
                "candidate_lamp_id": "candidate:gps:0007",
                "contributing_observations": ["obs_a", "obs_b"],
                "geo_summary": {"lat": 12.9716, "lon": 77.5946, "gps_accuracy_m": 4.0},
                "consensus_metrics": {"overall_useful_illumination_score": 0.84},
            }
        ]
    }
    path = tmp_path / "route_aggregation.json"
    path.write_text(json.dumps(route_aggregation), encoding="utf-8")

    store = LocationPriorStore.load(path)
    match = store.best_match(12.97161, 77.59461, 4.0, LocationPriorSettings())

    assert len(store.candidates) == 1
    assert match is not None
    assert match.candidate_lamp_id == "candidate:gps:0007"
    assert match.observation_count == 2
