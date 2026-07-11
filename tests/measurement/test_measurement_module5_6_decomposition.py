import json
from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_measurement.cli.train_module import main as train_module_main
from rbccps_measurement.contracts.input_schema import CameraMetadata, DetectorTrackRecord, FrameRecord, PoseRecord
from rbccps_measurement.contracts.module_io import AffectedRegionFieldOutput, NormalizedFrameProduct, SegmentationMaskOutput
from rbccps_measurement.decomposition.source_slots import SOURCE_CLASSES, deterministic_source_decomposition
from rbccps_measurement.decomposition.task_supervised_decomposition import RIS_INTERPRETATION, deterministic_ris_decomposition
from rbccps_measurement.pipeline import run_clip_to_directory
from rbccps_measurement.segmentation.illumination_disentangled import SEMANTIC_CLASSES


def make_frame(image_uri: str = "frames/000001.jpg", width: int = 64, height: int = 48) -> FrameRecord:
    return FrameRecord(
        frame_id=1,
        timestamp_ns=1000,
        image_uri=image_uri,
        image_format="jpg",
        width=width,
        height=height,
        camera=CameraMetadata(exposure_time_s=0.0167, sensor_sensitivity_iso=800, ae_mode="auto", metadata_quality="partial"),
        pose=PoseRecord(latitude=12.9716, longitude=77.5946, gps_accuracy_m=4.0, speed_mps=4.0, imu_quality="good"),
    )


def make_track() -> DetectorTrackRecord:
    return DetectorTrackRecord(
        frame_id=1,
        timestamp_ns=1000,
        track_id="lamp_1",
        class_name="streetlight_lamp_head",
        bbox_xyxy=(28, 8, 36, 18),
        bbox_format="pixel_xyxy_original_frame",
        detector_score=0.9,
        track_confidence=0.9,
        lost_count=0,
    )


def make_product(frame: FrameRecord, reliability: float = 0.85) -> NormalizedFrameProduct:
    shape = (frame.height, frame.width)
    luma = np.full(shape, 0.08, dtype=np.float32)
    luma[22:, 22:44] = 0.62
    luma[20:32, 4:18] = 0.74
    return NormalizedFrameProduct(
        frame_id=frame.frame_id,
        timestamp_ns=frame.timestamp_ns,
        width=frame.width,
        height=frame.height,
        luma_proxy=luma,
        reliability_mask=np.full(shape, reliability, dtype=np.float32),
        radiometric_uncertainty=np.full(shape, 1.0 - reliability, dtype=np.float32),
        saturation_mask=np.zeros(shape, dtype=bool),
        bloom_mask=np.zeros(shape, dtype=bool),
        glare_mask=np.zeros(shape, dtype=bool),
        exposure_factor=1.0,
        reliability_score=reliability,
    )


def make_segmentation(frame: FrameRecord, *, shop: bool = False, reflection: bool = False) -> SegmentationMaskOutput:
    shape = (frame.height, frame.width)
    semantic = {label: np.zeros(shape, dtype=np.float32) for label in SEMANTIC_CLASSES}
    semantic["road"][22:, :] = 1.0
    semantic["footpath"][22:, :16] = 0.8
    if shop:
        semantic["shopfront"][10:30, 4:20] = 1.0
        semantic["window"][12:28, 6:18] = 1.0
    if reflection:
        semantic["wet_reflection_like_road"][28:42, 22:46] = 1.0
    conf = np.maximum.reduce([semantic["shopfront"], semantic["window"], semantic["wet_reflection_like_road"]])
    return SegmentationMaskOutput(
        frame_id=frame.frame_id,
        semantic_masks=semantic,
        class_order=SEMANTIC_CLASSES,
        public_space_mask=np.maximum(semantic["road"], semantic["footpath"]),
        occluder_mask=np.zeros(shape, dtype=np.float32),
        confounder_mask=conf,
        confounder_candidate_mask=conf,
        uncertainty_map=np.full(shape, 0.1, dtype=np.float32),
        confidence=0.9,
    )


def make_affected(frame: FrameRecord, active: bool = True) -> AffectedRegionFieldOutput:
    shape = (frame.height, frame.width)
    public = np.zeros(shape, dtype=np.float32)
    public[22:, :] = 1.0
    affected = np.zeros(shape, dtype=np.float32)
    if active:
        affected[22:, 24:42] = 1.0
    return AffectedRegionFieldOutput(
        track_id="lamp_1",
        frame_id=frame.frame_id,
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


def make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    frames = clips / "frames"
    annotations = dataset / "annotations"
    frames.mkdir(parents=True)
    annotations.mkdir(parents=True)
    image = Image.new("RGB", (64, 48), (10, 10, 12))
    for x in range(28, 36):
        for y in range(8, 18):
            image.putpixel((x, y), (240, 230, 190))
    image.save(frames / "000001.jpg")
    clip = {
        "clip_id": "clip_decomp",
        "device_id": "phone_test",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "frames": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "image_uri": "frames/000001.jpg",
                "image_format": "jpg",
                "width": 64,
                "height": 48,
                "camera": {"exposure_time_s": 0.0167, "sensor_sensitivity_iso": 800, "ae_mode": "auto", "metadata_quality": "partial"},
                "pose": {"latitude": 12.9716, "longitude": 77.5946, "gps_accuracy_m": 4.0, "speed_mps": 4.0, "imu_quality": "good"},
            }
        ],
        "tracks": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "track_id": "lamp_1",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [28, 8, 36, 18],
                "bbox_format": "pixel_xyxy_original_frame",
                "detector_score": 0.9,
                "track_confidence": 0.9,
            }
        ],
        "optional_calibration": {"photometric": {}, "map_priors": {}},
    }
    (clips / "clip_decomp.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"dataset_type": "rbccps_measurement", "clips": [{"clip_id": "clip_decomp", "manifest": "clips/clip_decomp.json"}]}),
        encoding="utf-8",
    )
    (annotations / "confounders.csv").write_text("clip_id,frame_id,confounder_type\nclip_decomp,1,shopfront_or_window\n", encoding="utf-8")
    return dataset


def test_source_fields_are_bounded_and_reconstruct_luma_budget():
    frame = make_frame()
    product = make_product(frame)
    output = deterministic_source_decomposition(
        "lamp_1",
        [make_track()],
        {1: frame},
        normalized_product=product,
        segmentation=make_segmentation(frame, shop=True, reflection=True),
        affected_region=make_affected(frame),
        status_confidence=0.9,
    )

    assert set(output.source_fields) == set(SOURCE_CLASSES)
    for field in output.source_fields.values():
        assert field.shape == product.luma_proxy.shape
        assert float(np.min(field)) >= 0.0
        assert float(np.max(field)) <= 1.0
    reconstructed = np.sum(np.stack([output.source_fields[name] for name in SOURCE_CLASSES], axis=0), axis=0)
    assert float(np.mean(np.abs(reconstructed - product.luma_proxy))) <= 0.02
    assert output.reconstruction_error <= 0.02


def test_target_source_is_clipped_to_affected_public_space():
    frame = make_frame()
    output = deterministic_source_decomposition(
        "lamp_1",
        [make_track()],
        {1: frame},
        normalized_product=make_product(frame),
        segmentation=make_segmentation(frame),
        affected_region=make_affected(frame),
        status_confidence=0.9,
    )

    target = output.source_fields["target_lamp"]
    affected = make_affected(frame).affected_field
    assert np.max(target[affected <= 0.0]) == 0.0
    assert np.max(target[affected > 0.0]) > 0.0


def test_shopfront_and_reflection_masks_raise_matching_source_probabilities():
    frame = make_frame()
    base = deterministic_source_decomposition(
        "lamp_1",
        [make_track()],
        {1: frame},
        normalized_product=make_product(frame),
        segmentation=make_segmentation(frame, shop=False, reflection=False),
        affected_region=make_affected(frame, active=False),
        status_confidence=0.2,
    )
    confounded = deterministic_source_decomposition(
        "lamp_1",
        [make_track()],
        {1: frame},
        normalized_product=make_product(frame),
        segmentation=make_segmentation(frame, shop=True, reflection=True),
        affected_region=make_affected(frame, active=False),
        status_confidence=0.2,
    )

    assert confounded.source_probabilities["shopfront_or_window"] > base.source_probabilities["shopfront_or_window"]
    assert confounded.source_probabilities["reflection"] > base.source_probabilities["reflection"]
    assert confounded.confounder_penalty > base.confounder_penalty


def test_unknown_bright_residual_increases_unknown_source():
    frame = make_frame()
    product = make_product(frame)
    product.luma_proxy[:, :] = 0.9
    output = deterministic_source_decomposition(
        "lamp_1",
        [make_track()],
        {1: frame},
        normalized_product=product,
        segmentation=make_segmentation(frame),
        affected_region=make_affected(frame, active=False),
        status_confidence=0.1,
    )

    assert output.source_probabilities["unknown_bright_source"] > 0.45
    assert "target_lamp" in output.source_probabilities


def test_ris_fields_are_bounded_and_non_physical_metadata_is_explicit():
    frame = make_frame()
    product = make_product(frame)
    source = deterministic_source_decomposition(
        "lamp_1",
        [make_track()],
        {1: frame},
        normalized_product=product,
        segmentation=make_segmentation(frame, reflection=True),
        affected_region=make_affected(frame),
    )

    output = deterministic_ris_decomposition(frame, normalized_product=product, segmentation=make_segmentation(frame), source_output=source)

    assert output.reflectance_like.shape == (frame.height, frame.width, 3)
    assert output.illumination_like.shape == (frame.height, frame.width)
    assert output.source_like.shape == (frame.height, frame.width)
    assert output.reconstruction_proxy.shape == (frame.height, frame.width, 3)
    assert 0.0 <= float(np.min(output.reflectance_like)) <= float(np.max(output.reflectance_like)) <= 1.0
    assert 0.0 <= float(np.min(output.illumination_like)) <= float(np.max(output.illumination_like)) <= 1.0
    assert output.metadata["interpretation"] == RIS_INTERPRETATION
    assert output.metadata["physical_claim"] is False


def test_ris_confidence_drops_with_low_reliability():
    frame = make_frame()
    high = deterministic_ris_decomposition(frame, normalized_product=make_product(frame, reliability=0.95))
    low = deterministic_ris_decomposition(frame, normalized_product=make_product(frame, reliability=0.2))

    assert low.decomposition_confidence < high.decomposition_confidence


def test_train_module_source_and_ris_write_loadable_checkpoints(tmp_path: Path, monkeypatch):
    dataset = make_dataset(tmp_path)

    source_out = tmp_path / "source_train"
    monkeypatch.setattr(
        "sys.argv",
        ["train-module", "--module", "source_decomposition", "--dataset", str(dataset), "--out", str(source_out), "--skip-readiness"],
    )
    train_module_main()
    source_checkpoint = json.loads((source_out / "source_decomposition_checkpoint.json").read_text(encoding="utf-8"))
    assert source_checkpoint["label_maps"]["source"] == list(SOURCE_CLASSES)
    if source_checkpoint["weights"]:
        assert (source_out / source_checkpoint["weights"]).exists()

    ris_out = tmp_path / "ris_train"
    monkeypatch.setattr(
        "sys.argv",
        ["train-module", "--module", "ris_decomposition", "--dataset", str(dataset), "--out", str(ris_out), "--skip-readiness"],
    )
    train_module_main()
    ris_checkpoint = json.loads((ris_out / "ris_decomposition_checkpoint.json").read_text(encoding="utf-8"))
    assert ris_checkpoint["training_summary"]["interpretation"] == RIS_INTERPRETATION
    if ris_checkpoint["weights"]:
        assert (ris_out / ris_checkpoint["weights"]).exists()


def test_pipeline_uses_modules_5_6_without_report_schema_churn(tmp_path: Path):
    dataset = make_dataset(tmp_path)
    manifest = dataset / "clips" / "clip_decomp.json"

    report = run_clip_to_directory(manifest, tmp_path / "run")[0].to_dict()

    assert "confounder_penalty" in report["metrics"]
    assert report["traceability"]["model_versions"]["source_decomposition"] == "deterministic_source_slots_v1"
    assert report["traceability"]["model_versions"]["ris_decomposition"] == "deterministic_ris_decomposition_v1"
    assert "source_fields" not in report
    assert "ris_decomposition" not in report
