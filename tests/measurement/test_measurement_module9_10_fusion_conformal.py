import json
from pathlib import Path

from PIL import Image

from rbccps_measurement.attribution.counterfactual import AttributionEstimate
from rbccps_measurement.cli.train_module import main as train_module_main
from rbccps_measurement.features.distributional_coverage import UsefulIlluminationFeatures
from rbccps_measurement.fusion.conformal import build_calibration_group_key, decide_abstention
from rbccps_measurement.fusion.monotonic_heads import EDGE_TYPES, monotonic_fuse
from rbccps_measurement.pipeline import run_clip_to_directory


def make_features(**overrides) -> UsefulIlluminationFeatures:
    values = {
        "coverage_proxy": 0.55,
        "adequacy_proxy": 0.55,
        "adequacy_class": "marginal",
        "uniformity_proxy": 0.55,
        "dark_hole_fraction": 0.2,
        "glare_penalty": 0.1,
        "confounder_penalty": 0.1,
        "occlusion_penalty": 0.1,
        "temporal_stability": 0.7,
        "utility_score": 0.55,
    }
    values.update(overrides)
    return UsefulIlluminationFeatures(**values)


def make_attribution(**overrides) -> AttributionEstimate:
    values = {
        "score": 0.65,
        "attribution_class": "certain",
        "uncertainty": 0.15,
        "all_source_utility": 0.7,
        "without_target_utility": 0.2,
    }
    values.update(overrides)
    return AttributionEstimate(**values)


def make_context(**overrides):
    values = {
        "device_id": "phone_test",
        "route_group": "route_a",
        "capture_mode": "night_video",
        "metadata_quality_score": 0.85,
        "auto_exposure": 1.0,
        "geometry_quality": 0.9,
        "gps_quality": "good",
        "hdr_mode": "off",
        "night_mode": True,
    }
    values.update(overrides)
    return values


def make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    frames = clips / "frames"
    annotations = dataset / "annotations"
    frames.mkdir(parents=True)
    annotations.mkdir(parents=True)
    image = Image.new("RGB", (48, 32), (10, 10, 12))
    image.save(frames / "000001.jpg")
    clip = {
        "clip_id": "clip_fusion",
        "device_id": "phone_test",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "frames": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "image_uri": "frames/000001.jpg",
                "image_format": "jpg",
                "width": 48,
                "height": 32,
                "camera": {"exposure_time_s": 0.0167, "sensor_sensitivity_iso": 800, "ae_mode": "auto", "hdr_mode": "off", "night_mode": True, "metadata_quality": "partial"},
                "pose": {"latitude": 12.9716, "longitude": 77.5946, "gps_accuracy_m": 4.0, "imu_quality": "good"},
            }
        ],
        "tracks": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "track_id": "lamp_1",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [20, 6, 28, 14],
                "bbox_format": "pixel_xyxy_original_frame",
                "detector_score": 0.9,
                "track_confidence": 0.9,
            }
        ],
        "optional_calibration": {"photometric": {}, "map_priors": {"route_group": "route_a"}},
    }
    (clips / "clip_fusion.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"dataset_type": "rbccps_measurement", "clips": [{"clip_id": "clip_fusion", "manifest": "clips/clip_fusion.json"}]}),
        encoding="utf-8",
    )
    (annotations / "qa_flags.csv").write_text("clip_id,frame_id,track_id,flag\nclip_fusion,1,lamp_1,reviewed\n", encoding="utf-8")
    return dataset


def test_monotonic_score_increases_for_positive_evidence():
    base = monotonic_fuse(make_features(coverage_proxy=0.4), make_attribution(), 0.9, track_id="lamp_1", context=make_context())
    higher_coverage = monotonic_fuse(make_features(coverage_proxy=0.7), make_attribution(), 0.9, track_id="lamp_1", context=make_context())
    higher_uniformity = monotonic_fuse(make_features(uniformity_proxy=0.8), make_attribution(), 0.9, track_id="lamp_1", context=make_context())
    higher_attribution = monotonic_fuse(make_features(), make_attribution(score=0.9), 0.9, track_id="lamp_1", context=make_context())

    assert higher_coverage.overall_score >= base.overall_score
    assert higher_uniformity.overall_score >= base.overall_score
    assert higher_attribution.overall_score >= base.overall_score


def test_monotonic_score_does_not_increase_for_penalties():
    base = monotonic_fuse(make_features(), make_attribution(uncertainty=0.1), 0.9, track_id="lamp_1", context=make_context())
    glare = monotonic_fuse(make_features(glare_penalty=0.8), make_attribution(uncertainty=0.1), 0.9, track_id="lamp_1", context=make_context())
    confounder = monotonic_fuse(make_features(confounder_penalty=0.8), make_attribution(uncertainty=0.1), 0.9, track_id="lamp_1", context=make_context())
    occlusion = monotonic_fuse(make_features(occlusion_penalty=0.8), make_attribution(uncertainty=0.1), 0.9, track_id="lamp_1", context=make_context())
    uncertain = monotonic_fuse(make_features(), make_attribution(uncertainty=0.8), 0.9, track_id="lamp_1", context=make_context())

    assert glare.overall_score <= base.overall_score
    assert confounder.overall_score <= base.overall_score
    assert occlusion.overall_score <= base.overall_score
    assert uncertain.overall_score <= base.overall_score
    assert uncertain.confidence < base.confidence


def test_scene_graph_contains_required_nodes_and_edges():
    fusion = monotonic_fuse(make_features(), make_attribution(), 0.9, track_id="lamp_1", region_mix={"road": 1.0}, context=make_context())
    graph = fusion.fusion_output.graph

    node_types = {node["type"] for node in graph.nodes}
    edge_types = {edge["type"] for edge in graph.edges}
    assert {"lamp", "region", "confounder", "camera", "geometry", "context"}.issubset(node_types)
    assert set(EDGE_TYPES).issubset(edge_types)
    assert fusion.fusion_output.metadata["monotonic_constraints"]["score_positive"]


def test_calibration_group_key_is_deterministic_and_low_risk_reports():
    context = make_context()
    key_a = build_calibration_group_key(context, 0.1).to_key()
    key_b = build_calibration_group_key(context, 0.1).to_key()
    fusion = monotonic_fuse(make_features(), make_attribution(), 0.95, track_id="lamp_1", context=context)

    decision = decide_abstention(fusion.overall_category, fusion.confidence, [], fusion_output=fusion.fusion_output, context=context)

    assert key_a == key_b
    assert decision.action == "report"
    assert decision.calibration_output.calibration_group_key == key_a
    assert decision.prediction_set


def test_low_confidence_widens_prediction_set_and_high_risk_abstains():
    risky_context = make_context(device_id="unknown_device", route_group="unknown_route", gps_quality="missing", hdr_mode="unknown")
    fusion = monotonic_fuse(
        make_features(confounder_penalty=0.9, dark_hole_fraction=0.8, occlusion_penalty=0.7),
        make_attribution(uncertainty=0.9),
        0.35,
        track_id="lamp_1",
        context=risky_context,
    )

    decision = decide_abstention(fusion.overall_category, fusion.confidence, ["high_fusion_uncertainty"], fusion_output=fusion.fusion_output, context=risky_context)

    assert decision.action == "abstain"
    assert "manual_review_recommended" in decision.prediction_set
    assert decision.calibration_output.risk_estimate > 0.68


def test_train_module_fusion_and_conformal_write_checkpoints(tmp_path: Path, monkeypatch):
    dataset = make_dataset(tmp_path)

    fusion_out = tmp_path / "fusion_train"
    monkeypatch.setattr("sys.argv", ["train-module", "--module", "fusion", "--dataset", str(dataset), "--out", str(fusion_out), "--skip-readiness"])
    train_module_main()
    fusion_checkpoint = json.loads((fusion_out / "fusion_checkpoint.json").read_text(encoding="utf-8"))
    assert fusion_checkpoint["monotonic_constraints"]["positive"]
    if fusion_checkpoint["weights"]:
        assert (fusion_out / fusion_checkpoint["weights"]).exists()

    conformal_out = tmp_path / "conformal_train"
    monkeypatch.setattr("sys.argv", ["train-module", "--module", "conformal", "--dataset", str(dataset), "--out", str(conformal_out), "--skip-readiness"])
    train_module_main()
    conformal_checkpoint = json.loads((conformal_out / "conformal_checkpoint.json").read_text(encoding="utf-8"))
    assert conformal_checkpoint["training_summary"]["group_key_schema"]
    if conformal_checkpoint["weights"]:
        assert (conformal_out / conformal_checkpoint["weights"]).exists()


def test_pipeline_uses_module9_10_traceability_and_stable_report_schema(tmp_path: Path):
    dataset = make_dataset(tmp_path)
    report = run_clip_to_directory(dataset / "clips" / "clip_fusion.json", tmp_path / "run")[0].to_dict()

    assert "overall_useful_illumination_score" in report["metrics"]
    assert "overall_category" in report["metrics"]
    assert "overall" in report["confidence"]
    assert "prediction_set" in report["confidence"]
    assert "action" in report["confidence"]
    assert report["traceability"]["model_versions"]["fusion"] == "deterministic_monotonic_scene_graph_fusion_v1"
    assert report["traceability"]["model_versions"]["conformal"] == "deterministic_group_conformal_abstention_v1"
