import json
from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_measurement.attribution.counterfactual import estimate_counterfactual_attribution
from rbccps_measurement.cli.train_module import main as train_module_main
from rbccps_measurement.contracts.input_schema import CameraMetadata, DetectorTrackRecord, FrameRecord, PoseRecord
from rbccps_measurement.contracts.module_io import AffectedRegionFieldOutput, RISDecompositionOutput, SegmentationMaskOutput, SourceFieldOutput
from rbccps_measurement.decomposition.source_slots import SOURCE_CLASSES, SourceEvidence
from rbccps_measurement.features.distributional_coverage import estimate_useful_features
from rbccps_measurement.geometry.lamp_footprint_field import FootprintEstimate
from rbccps_measurement.pipeline import run_clip_to_directory
from rbccps_measurement.segmentation.illumination_disentangled import SEMANTIC_CLASSES
from rbccps_measurement.status.latent_emission_state import LampStatusEstimate


def make_frame() -> FrameRecord:
    return FrameRecord(
        frame_id=1,
        timestamp_ns=1000,
        image_uri="frames/000001.jpg",
        image_format="jpg",
        width=48,
        height=32,
        camera=CameraMetadata(exposure_time_s=0.0167, sensor_sensitivity_iso=800, ae_mode="auto", metadata_quality="partial"),
        pose=PoseRecord(latitude=12.9716, longitude=77.5946, gps_accuracy_m=4.0, imu_quality="good"),
    )


def make_status() -> LampStatusEstimate:
    return LampStatusEstimate(
        label="on",
        confidence=0.86,
        dim_probability=0.05,
        occluded_probability=0.1,
        flicker_index=0.05,
        saturated_flag=False,
    )


def make_affected(active: bool = True) -> AffectedRegionFieldOutput:
    shape = (32, 48)
    public = np.zeros(shape, dtype=np.float32)
    public[12:, :] = 1.0
    affected = np.zeros(shape, dtype=np.float32)
    if active:
        affected[12:, 12:40] = 1.0
    return AffectedRegionFieldOutput(
        track_id="lamp_1",
        frame_id=1,
        affected_field=affected,
        public_space_mask=public,
        road_region=affected.copy(),
        footpath_region=np.zeros(shape, dtype=np.float32),
        crossing_region=np.zeros(shape, dtype=np.float32),
        verge_region=np.zeros(shape, dtype=np.float32),
        occlusion_gate=np.ones(shape, dtype=np.float32),
        uncertainty_map=np.full(shape, 0.1, dtype=np.float32),
        region_mix={"road": 1.0, "footpath": 0.0, "crossing": 0.0, "verge": 0.0},
        mask_ref="masks/lamp_1.json",
        quality="good",
        geometry_quality=1.0,
        field_confidence=0.9,
    )


def make_segmentation() -> SegmentationMaskOutput:
    shape = (32, 48)
    semantic = {label: np.zeros(shape, dtype=np.float32) for label in SEMANTIC_CLASSES}
    semantic["road"][12:, :] = 1.0
    return SegmentationMaskOutput(
        frame_id=1,
        semantic_masks=semantic,
        class_order=SEMANTIC_CLASSES,
        public_space_mask=semantic["road"],
        occluder_mask=np.zeros(shape, dtype=np.float32),
        confounder_mask=np.zeros(shape, dtype=np.float32),
        confounder_candidate_mask=np.zeros(shape, dtype=np.float32),
        uncertainty_map=np.full(shape, 0.1, dtype=np.float32),
        confidence=0.9,
    )


def make_ris(uneven: bool = True) -> RISDecompositionOutput:
    shape = (32, 48)
    illumination = np.full(shape, 0.5, dtype=np.float32)
    if uneven:
        illumination[12:22, 12:26] = 0.08
        illumination[22:, 26:40] = 0.8
    source = np.clip(illumination - 0.25, 0.0, 1.0).astype(np.float32)
    rgb = np.repeat(illumination[:, :, None], 3, axis=2)
    return RISDecompositionOutput(
        frame_id=1,
        reflectance_like=rgb,
        illumination_like=illumination,
        source_like=source,
        reconstruction_proxy=rgb,
        confidence_map=np.full(shape, 0.85, dtype=np.float32),
        reconstruction_error=0.04,
        decomposition_confidence=0.84,
    )


def make_source(target: float = 0.7, competitor: float = 0.1) -> SourceFieldOutput:
    shape = (32, 48)
    fields = {name: np.zeros(shape, dtype=np.float32) for name in SOURCE_CLASSES}
    fields["target_lamp"][12:, 12:40] = target
    fields["shopfront_or_window"][12:, 12:40] = competitor
    fields["unknown_bright_source"][:, :] = 0.02
    probs = {name: float(np.sum(field)) for name, field in fields.items()}
    total = sum(probs.values())
    probs = {name: value / total for name, value in probs.items()}
    return SourceFieldOutput(
        frame_id=1,
        track_id="lamp_1",
        source_fields=fields,
        source_probabilities=probs,
        residual_field=np.zeros(shape, dtype=np.float32),
        reconstruction_error=0.01,
        confounder_penalty=probs["shopfront_or_window"] + 0.5 * probs["unknown_bright_source"],
        source_confusion_score=probs["shopfront_or_window"],
    )


def make_footprint() -> FootprintEstimate:
    field = make_affected()
    return FootprintEstimate(quality="good", mask_ref=field.mask_ref, geometry_quality=1.0, field=field)


def make_source_evidence(output: SourceFieldOutput) -> SourceEvidence:
    p = output.source_probabilities
    return SourceEvidence(
        target_lamp=p["target_lamp"],
        other_lamps=p["other_lamp"],
        headlights=p["headlight"],
        shopfronts=p["shopfront_or_window"],
        reflections=p["reflection"],
        unknown=p["unknown_bright_source"],
        sign_or_signal=p["sign_or_signal"],
        field_output=output,
    )


def make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    frames = clips / "frames"
    annotations = dataset / "annotations"
    frames.mkdir(parents=True)
    annotations.mkdir(parents=True)
    Image.new("RGB", (48, 32), (12, 12, 14)).save(frames / "000001.jpg")
    clip = {
        "clip_id": "clip_feat_attr",
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
                "camera": {"exposure_time_s": 0.0167, "sensor_sensitivity_iso": 800, "ae_mode": "auto", "metadata_quality": "partial"},
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
        "optional_calibration": {"photometric": {}, "map_priors": {}},
    }
    (clips / "clip_feat_attr.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"dataset_type": "rbccps_measurement", "clips": [{"clip_id": "clip_feat_attr", "manifest": "clips/clip_feat_attr.json"}]}),
        encoding="utf-8",
    )
    (annotations / "affected_regions.csv").write_text("clip_id,frame_id,track_id,region_type\nclip_feat_attr,1,lamp_1,affected_road\n", encoding="utf-8")
    (annotations / "visibility_labels.csv").write_text("clip_id,frame_id,track_id,visibility\nclip_feat_attr,1,lamp_1,good\n", encoding="utf-8")
    (annotations / "attribution_labels.csv").write_text("clip_id,frame_id,track_id,attribution_class\nclip_feat_attr,1,lamp_1,certain\n", encoding="utf-8")
    return dataset


def test_distributional_features_emit_quantiles_dark_holes_and_region_quality():
    source = make_source()
    features = estimate_useful_features(
        make_status(),
        make_footprint(),
        make_source_evidence(source),
        0.9,
        segmentation=make_segmentation(),
        ris_output=make_ris(uneven=True),
        source_output=source,
    )

    assert set(features.quantiles or {}) == {"q10", "q25", "q50", "q75", "q90"}
    assert features.quantiles["q10"] <= features.quantiles["q50"] <= features.quantiles["q90"]
    assert features.dark_hole_fraction > 0.0
    assert features.dark_hole_probability_map.shape == (32, 48)
    assert features.quality_by_region["road"] >= 0.0
    assert "illumination_q50" in features.to_dict()


def test_distributional_uniformity_is_higher_for_even_illumination():
    source = make_source()
    uneven = estimate_useful_features(make_status(), make_footprint(), make_source_evidence(source), 0.9, ris_output=make_ris(True), source_output=source)
    even = estimate_useful_features(make_status(), make_footprint(), make_source_evidence(source), 0.9, ris_output=make_ris(False), source_output=source)

    assert even.uniformity_proxy > uneven.uniformity_proxy
    assert uneven.dark_hole_fraction > even.dark_hole_fraction


def test_counterfactual_attribution_is_certain_when_target_collapse_is_large():
    source = make_source(target=0.8, competitor=0.02)
    features = estimate_useful_features(make_status(), make_footprint(), make_source_evidence(source), 0.9, ris_output=make_ris(False), source_output=source)

    attribution = estimate_counterfactual_attribution(features, make_source_evidence(source), affected_region=make_affected(), source_output=source)

    assert attribution.all_source_utility > attribution.without_target_utility
    assert attribution.score > 0.55
    assert attribution.attribution_class == "certain"


def test_counterfactual_attribution_becomes_mixed_when_competitor_dominates():
    source = make_source(target=0.05, competitor=0.85)
    features = estimate_useful_features(make_status(), make_footprint(), make_source_evidence(source), 0.9, ris_output=make_ris(False), source_output=source)

    attribution = estimate_counterfactual_attribution(features, make_source_evidence(source), affected_region=make_affected(), source_output=source)

    assert attribution.score < 0.45
    assert attribution.attribution_class == "mixed"
    assert attribution.source_competition["shopfront_or_window"] > attribution.source_competition["target_lamp"]


def test_train_module_features_and_attribution_write_checkpoints(tmp_path: Path, monkeypatch):
    dataset = make_dataset(tmp_path)

    feature_out = tmp_path / "feature_train"
    monkeypatch.setattr("sys.argv", ["train-module", "--module", "features", "--dataset", str(dataset), "--out", str(feature_out), "--skip-readiness"])
    train_module_main()
    feature_checkpoint = json.loads((feature_out / "features_checkpoint.json").read_text(encoding="utf-8"))
    assert feature_checkpoint["label_maps"]["quantiles"] == ["q10", "q25", "q50", "q75", "q90"]
    if feature_checkpoint["weights"]:
        assert (feature_out / feature_checkpoint["weights"]).exists()

    attribution_out = tmp_path / "attribution_train"
    monkeypatch.setattr("sys.argv", ["train-module", "--module", "attribution", "--dataset", str(dataset), "--out", str(attribution_out), "--skip-readiness"])
    train_module_main()
    attribution_checkpoint = json.loads((attribution_out / "attribution_checkpoint.json").read_text(encoding="utf-8"))
    assert attribution_checkpoint["label_maps"]["attribution"] == ["certain", "mixed", "uncertain"]
    if attribution_checkpoint["weights"]:
        assert (attribution_out / attribution_checkpoint["weights"]).exists()


def test_pipeline_uses_module7_8_traceability_without_report_schema_churn(tmp_path: Path):
    dataset = make_dataset(tmp_path)
    report = run_clip_to_directory(dataset / "clips" / "clip_feat_attr.json", tmp_path / "run")[0].to_dict()

    assert report["traceability"]["model_versions"]["features"] == "deterministic_distributional_features_v1"
    assert report["traceability"]["model_versions"]["attribution"] == "deterministic_counterfactual_attribution_v1"
    assert "illumination_q50" in report["metrics"]
    assert "attribution" not in report
