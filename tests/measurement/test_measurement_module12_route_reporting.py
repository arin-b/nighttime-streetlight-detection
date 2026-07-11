import json
from pathlib import Path

from rbccps_measurement.cli.measure_batch import main as measure_batch_main
from rbccps_measurement.cli.train_module import main as train_module_main
from rbccps_measurement.contracts.output_schema import MeasurementReport
from rbccps_measurement.route.graph_aggregation import EDGE_TYPES, aggregate_route_reports

from test_measurement_contracts import make_manifest_payload


def make_report(
    observation_id: str,
    clip_id: str,
    lat: float | None = 12.9716,
    lon: float | None = 77.5946,
    score: float = 0.6,
    category: str = "marginal",
    confidence: float = 0.8,
    mapped_lamp_id: str | None = None,
    physical_valid: bool = False,
) -> MeasurementReport:
    return MeasurementReport(
        measurement_run_id="run",
        lamp_observation_id=observation_id,
        lamp_track_id=f"track_{observation_id}",
        mapped_lamp_id=mapped_lamp_id,
        clip_id=clip_id,
        time_window={"start_ns": 1, "end_ns": 2, "num_frames_used": 1, "evidence_frames": [1]},
        geo_summary={"lat": lat, "lon": lon, "gps_accuracy_m": 4.0 if lat is not None else None},
        status={"label": "on", "confidence": 0.9},
        affected_region={"quality": "good", "image_mask_uri": f"masks/{observation_id}.json", "region_mix": {"road": 1.0}},
        metrics={"overall_useful_illumination_score": score, "overall_category": category},
        confidence={"overall": confidence, "calibration_level": 1, "attribution_class": "certain", "action": "report"},
        uncertainty_flags=[],
        optional_physical_estimates={
            "valid": physical_valid,
            "reason": "test",
            "horizontal_illuminance_lux_mean": 4.0 if physical_valid else None,
            "horizontal_illuminance_lux_interval": [2.5, 6.5] if physical_valid else None,
            "vertical_illuminance_lux_mean": None,
            "served_area_m2_est": None,
        },
        traceability={
            "model_versions": {"pipeline": "test_pipeline", "photometry": "test_photometry"},
            "feature_snapshot_ref": f"features/{observation_id}.json",
            "policy_id": "policy",
        },
    )


def test_route_graph_contains_required_nodes_and_edges():
    reports = [
        make_report("obs_a", "clip_a", mapped_lamp_id="lamp_001"),
        make_report("obs_b", "clip_b", lat=12.97162, lon=77.59462, mapped_lamp_id="lamp_001"),
    ]
    output = aggregate_route_reports(reports, source_report_refs={"obs_a": "clip_a/reports.json"}, route_group="route_a")

    node_types = {node["type"] for node in output.graph.nodes}
    edge_types = {edge["type"] for edge in output.graph.edges}

    assert {"observation", "candidate_lamp", "drive_pass", "gps_neighborhood", "road_segment", "map_prior"}.issubset(node_types)
    assert set(EDGE_TYPES).issubset(edge_types)
    assert len(output.lamps) == 1
    assert output.audit_trail.module_versions["pipeline"] == "test_pipeline"
    assert output.audit_trail.evidence_refs[0]["feature_snapshot_ref"] == "features/obs_a.json"


def test_gps_proximity_merges_nearby_but_not_far_observations():
    nearby = [
        make_report("obs_a", "clip_a", lat=12.9716, lon=77.5946),
        make_report("obs_b", "clip_b", lat=12.97162, lon=77.59462),
    ]
    far = [*nearby, make_report("obs_c", "clip_c", lat=12.9816, lon=77.6046)]

    nearby_output = aggregate_route_reports(nearby, route_group="route_a")
    far_output = aggregate_route_reports(far, route_group="route_a")

    assert len(nearby_output.lamps) == 1
    assert len(far_output.lamps) == 2


def test_disagreement_preserves_outlier_and_review_flags():
    reports = [
        make_report("obs_a", "clip_a", score=0.82, category="adequate", confidence=0.9),
        make_report("obs_b", "clip_b", lat=12.97162, lon=77.59462, score=0.12, category="unknown", confidence=0.9),
    ]
    output = aggregate_route_reports(reports, route_group="route_a")
    lamp = output.lamps[0]

    assert lamp.disagreement_score >= 0.42
    assert "route_category_disagreement" in lamp.quality_flags
    assert lamp.manual_review_priority == "high"
    assert lamp.consensus_metrics["worst_credible_category"] == "unknown"


def test_physical_estimates_aggregate_only_from_valid_reports():
    reports = [
        make_report("obs_a", "clip_a", physical_valid=True),
        make_report("obs_b", "clip_b", lat=12.97162, lon=77.59462, physical_valid=False),
    ]
    output = aggregate_route_reports(reports, route_group="route_a")
    physical = output.lamps[0].physical_estimate_summary

    assert physical["valid"] is True
    assert physical["valid_observations"] == 1
    assert physical["horizontal_illuminance_lux_mean"] == 4.0


def test_measure_batch_writes_module12_sidecars(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    clips.mkdir(parents=True)
    payload_a = make_manifest_payload(clip_id="clip_a")
    payload_b = make_manifest_payload(clip_id="clip_b")
    payload_b["frames"][0]["pose"]["latitude"] = 12.97162
    payload_b["frames"][0]["pose"]["longitude"] = 77.59462
    (clips / "clip_a.json").write_text(json.dumps(payload_a, indent=2), encoding="utf-8")
    (clips / "clip_b.json").write_text(json.dumps(payload_b, indent=2), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_type": "rbccps_measurement",
                "route_group": "route_a",
                "clips": [
                    {"clip_id": "clip_a", "manifest": "clips/clip_a.json"},
                    {"clip_id": "clip_b", "manifest": "clips/clip_b.json"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "batch"

    monkeypatch.setattr("sys.argv", ["measure-batch", "--dataset", str(dataset), "--out", str(out)])
    measure_batch_main()

    summary = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
    route = json.loads((out / "route_aggregation.json").read_text(encoding="utf-8"))
    assert summary["route_aggregation"]["candidate_lamps"] >= 1
    assert (out / "route_lamps.geojson").exists()
    assert (out / "road_segments.geojson").exists()
    assert (out / "audit_trail.json").exists()
    assert route["graph"]["provenance"]["implementation"] == "deterministic_route_graph_aggregation_v1"


def test_train_module_route_aggregation_writes_checkpoint(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset_manifest.json").write_text(json.dumps({"clips": []}), encoding="utf-8")
    out = tmp_path / "train"

    monkeypatch.setattr("sys.argv", ["train-module", "--module", "route_aggregation", "--dataset", str(dataset), "--out", str(out), "--skip-readiness"])
    train_module_main()

    checkpoint = json.loads((out / "route_aggregation_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["module"] == "route_aggregation"
    assert "observed_as" in checkpoint["graph_schema"]["edge_types"]
    if checkpoint["weights"]:
        assert (out / checkpoint["weights"]).exists()
