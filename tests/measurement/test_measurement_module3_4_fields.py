import json
from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_measurement.cli.train_module import main as train_module_main
from rbccps_measurement.contracts.input_schema import CameraMetadata, DetectorTrackRecord, FrameRecord, PoseRecord
from rbccps_measurement.contracts.module_io import NormalizedFrameProduct, SegmentationMaskOutput
from rbccps_measurement.geometry.lamp_footprint_field import estimate_affected_region_field
from rbccps_measurement.pipeline import run_clip_to_directory
from rbccps_measurement.segmentation.illumination_disentangled import SEMANTIC_CLASSES, deterministic_segment_frame


def make_frame(image_uri: str = "frames/000001.jpg", width: int = 96, height: int = 64, pose_good: bool = True) -> FrameRecord:
    pose = PoseRecord(latitude=12.9716, longitude=77.5946, gps_accuracy_m=4.0, heading_deg=80.0, imu_quality="good") if pose_good else PoseRecord()
    return FrameRecord(
        frame_id=1,
        timestamp_ns=1000,
        image_uri=image_uri,
        image_format="jpg",
        width=width,
        height=height,
        camera=CameraMetadata(exposure_time_s=0.0167, sensor_sensitivity_iso=800, ae_mode="auto", metadata_quality="partial"),
        pose=pose,
    )


def make_track(frame_id: int = 1, bbox=(42, 12, 52, 24)) -> DetectorTrackRecord:
    return DetectorTrackRecord(
        frame_id=frame_id,
        timestamp_ns=1000,
        track_id="lamp_1",
        class_name="streetlight_lamp_head",
        bbox_xyxy=bbox,
        bbox_format="pixel_xyxy_original_frame",
        detector_score=0.9,
        track_confidence=0.9,
        lost_count=0,
    )


def make_product(frame: FrameRecord, saturated: bool = False) -> NormalizedFrameProduct:
    shape = (frame.height, frame.width)
    saturation = np.zeros(shape, dtype=bool)
    glare = np.zeros(shape, dtype=bool)
    if saturated:
        saturation[20:28, 40:56] = True
        glare[20:32, 36:60] = True
    return NormalizedFrameProduct(
        frame_id=frame.frame_id,
        timestamp_ns=frame.timestamp_ns,
        width=frame.width,
        height=frame.height,
        luma_proxy=np.full(shape, 0.08, dtype=np.float32),
        reliability_mask=np.full(shape, 0.75, dtype=np.float32),
        radiometric_uncertainty=np.full(shape, 0.25, dtype=np.float32),
        saturation_mask=saturation,
        bloom_mask=np.zeros(shape, dtype=bool),
        glare_mask=glare,
        exposure_factor=1.0,
        reliability_score=0.75,
    )


def make_segmentation(frame: FrameRecord, occluded: bool = False) -> SegmentationMaskOutput:
    shape = (frame.height, frame.width)
    public = np.zeros(shape, dtype=np.float32)
    public[28:, :] = 1.0
    road = public.copy()
    footpath = np.zeros(shape, dtype=np.float32)
    footpath[28:, :20] = 1.0
    crossing = np.zeros(shape, dtype=np.float32)
    verge = np.zeros(shape, dtype=np.float32)
    occluder = np.zeros(shape, dtype=np.float32)
    if occluded:
        occluder[30:46, 38:58] = 1.0
    semantic = {label: np.zeros(shape, dtype=np.float32) for label in SEMANTIC_CLASSES}
    semantic.update({"road": road, "footpath": footpath, "crossing": crossing, "verge": verge, "occluder": occluder})
    return SegmentationMaskOutput(
        frame_id=frame.frame_id,
        semantic_masks=semantic,
        class_order=SEMANTIC_CLASSES,
        public_space_mask=public,
        occluder_mask=occluder,
        confounder_mask=np.zeros(shape, dtype=np.float32),
        confounder_candidate_mask=np.zeros(shape, dtype=np.float32),
        uncertainty_map=np.full(shape, 0.1, dtype=np.float32),
        confidence=0.9,
    )


def make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    clips = dataset / "clips"
    frames = clips / "frames"
    annotations = dataset / "annotations"
    frames.mkdir(parents=True)
    annotations.mkdir(parents=True)
    Image.new("RGB", (96, 64), (10, 10, 12)).save(frames / "000001.jpg")
    clip = {
        "clip_id": "clip_seg_footprint",
        "device_id": "phone_test",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "frames": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "image_uri": "frames/000001.jpg",
                "image_format": "jpg",
                "width": 96,
                "height": 64,
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
                "bbox_xyxy": [42, 12, 52, 24],
                "bbox_format": "pixel_xyxy_original_frame",
                "detector_score": 0.9,
                "track_confidence": 0.9,
            }
        ],
        "optional_calibration": {"photometric": {}, "map_priors": {}},
    }
    (clips / "clip_seg_footprint.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"dataset_type": "rbccps_measurement", "clips": [{"clip_id": "clip_seg_footprint", "manifest": "clips/clip_seg_footprint.json"}]}),
        encoding="utf-8",
    )
    (annotations / "public_space_regions.csv").write_text("clip_id,frame_id,region_type\nclip_seg_footprint,1,road\n", encoding="utf-8")
    (annotations / "affected_regions.csv").write_text("clip_id,frame_id,track_id,region_type\nclip_seg_footprint,1,lamp_1,affected_road\n", encoding="utf-8")
    return dataset


def test_segmentation_keeps_dark_public_space_structural_prior(tmp_path: Path):
    frame_root = tmp_path / "clip"
    (frame_root / "frames").mkdir(parents=True)
    Image.new("RGB", (96, 64), (2, 2, 3)).save(frame_root / "frames" / "000001.jpg")

    output = deterministic_segment_frame(make_frame(), frame_root, enhanced_image_path=frame_root / "enhanced.jpg")

    assert output.public_space_mask[48:, 40:56].mean() > 0.2
    assert output.semantic_masks["road"][48:, 40:56].mean() > 0.2
    assert output.enhanced_view_used is True
    assert output.metadata["enhanced_view_policy"] == "auxiliary_only"
    assert "road" in output.class_order


def test_segmentation_saturation_raises_confounder_and_uncertainty_without_erasing_road():
    frame = make_frame()
    output = deterministic_segment_frame(frame, normalized_product=make_product(frame, saturated=True))

    assert output.confounder_candidate_mask[20:32, 36:60].mean() > 0.25
    assert output.uncertainty_map[20:32, 36:60].mean() > output.uncertainty_map[0:8, 0:8].mean()
    assert output.public_space_mask[48:, 40:56].mean() > 0.2


def test_affected_field_is_zero_outside_public_space_and_reports_mix():
    frame = make_frame()
    track = make_track()
    segmentation = make_segmentation(frame)

    field = estimate_affected_region_field("lamp_1", [track], {1: frame}, {1: segmentation})

    assert np.max(field.affected_field[field.public_space_mask <= 0.0]) == 0.0
    assert field.affected_field[field.public_space_mask > 0.0].max() > 0.0
    assert abs(sum(field.region_mix.values()) - 1.0) < 0.001
    assert field.quality == "good"
    assert field.geometry_quality == 1.0


def test_occlusion_gate_reduces_affected_region_strength():
    frame = make_frame()
    track = make_track()
    clear = estimate_affected_region_field("lamp_1", [track], {1: frame}, {1: make_segmentation(frame, occluded=False)})
    occluded = estimate_affected_region_field("lamp_1", [track], {1: frame}, {1: make_segmentation(frame, occluded=True)})

    occluded_patch = np.s_[30:46, 38:58]
    assert occluded.occlusion_gate[occluded_patch].mean() < clear.occlusion_gate[occluded_patch].mean()
    assert occluded.affected_field[occluded_patch].mean() < clear.affected_field[occluded_patch].mean()


def test_weak_geometry_raises_quality_flag_and_uncertainty():
    weak_frame = make_frame(pose_good=False)
    field = estimate_affected_region_field("lamp_1", [make_track()], {1: weak_frame}, {1: make_segmentation(weak_frame)})

    assert field.quality == "weak"
    assert field.geometry_quality == 0.0
    assert "weak_geometry" in field.quality_flags
    assert field.uncertainty_map.mean() > 0.35


def test_train_module_segmentation_and_footprint_write_checkpoints(tmp_path: Path, monkeypatch):
    dataset = make_dataset(tmp_path)

    seg_out = tmp_path / "seg_train"
    monkeypatch.setattr(
        "sys.argv",
        ["train-module", "--module", "segmentation", "--dataset", str(dataset), "--out", str(seg_out), "--skip-readiness"],
    )
    train_module_main()
    seg_checkpoint = json.loads((seg_out / "segmentation_checkpoint.json").read_text(encoding="utf-8"))
    assert seg_checkpoint["label_maps"]["semantic"] == list(SEMANTIC_CLASSES)
    assert seg_checkpoint["fallback"] == "deterministic_illumination_disentangled_v1"

    footprint_out = tmp_path / "footprint_train"
    monkeypatch.setattr(
        "sys.argv",
        ["train-module", "--module", "footprint", "--dataset", str(dataset), "--out", str(footprint_out), "--skip-readiness"],
    )
    train_module_main()
    footprint_checkpoint = json.loads((footprint_out / "footprint_checkpoint.json").read_text(encoding="utf-8"))
    assert footprint_checkpoint["label_maps"]["region_mix"] == ["road", "footpath", "crossing", "verge"]
    assert footprint_checkpoint["fallback"] == "deterministic_lamp_conditioned_field_v1"


def test_pipeline_uses_module3_4_artifacts_in_report(tmp_path: Path):
    dataset = make_dataset(tmp_path)
    manifest = dataset / "clips" / "clip_seg_footprint.json"
    out = tmp_path / "run"

    report = run_clip_to_directory(manifest, out)[0].to_dict()
    mask_payload = json.loads((out / report["affected_region"]["image_mask_uri"]).read_text(encoding="utf-8"))

    assert report["traceability"]["model_versions"]["segmentation"] == "deterministic_illumination_disentangled_v1"
    assert report["traceability"]["model_versions"]["footprint"] == "deterministic_lamp_conditioned_field_v1"
    assert abs(sum(report["affected_region"]["region_mix"].values()) - 1.0) < 0.001
    assert mask_payload["module"] == "lamp_conditioned_affected_region_field"
