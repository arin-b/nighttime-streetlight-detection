import json
from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_measurement.cli.train_module import main as train_module_main
from rbccps_measurement.contracts.input_schema import CameraMetadata, DetectorTrackRecord, FrameRecord, PoseRecord
from rbccps_measurement.contracts.module_io import LampCropSequence, NormalizedFrameProduct
from rbccps_measurement.status.crop_sequence import TOKEN_NAMES, CropSequenceConfig, build_lamp_crop_sequence
from rbccps_measurement.status.model import StatusEstimator, deterministic_latent_status


def make_frame(image_uri: str = "frames/000001.jpg", width: int = 80, height: int = 60) -> FrameRecord:
    return FrameRecord(
        frame_id=1,
        timestamp_ns=1000,
        image_uri=image_uri,
        image_format="jpg",
        width=width,
        height=height,
        camera=CameraMetadata(
            exposure_time_s=0.0167,
            sensor_sensitivity_iso=800,
            ae_mode="auto",
            metadata_quality="partial",
        ),
        pose=PoseRecord(imu_quality="missing"),
    )


def make_track(
    bbox: tuple[float, float, float, float] = (30, 20, 40, 32),
    bbox_format: str = "pixel_xyxy_original_frame",
    frame_id: int = 1,
    lost_count: int = 0,
) -> DetectorTrackRecord:
    return DetectorTrackRecord(
        frame_id=frame_id,
        timestamp_ns=1000 + frame_id,
        track_id="lamp_1",
        class_name="streetlight_lamp_head",
        bbox_xyxy=bbox,
        bbox_format=bbox_format,
        detector_score=0.86,
        track_confidence=0.9,
        track_age=3,
        lost_count=lost_count,
    )


def make_product(frame: FrameRecord, reliability: float = 0.82) -> NormalizedFrameProduct:
    shape = (frame.height, frame.width)
    return NormalizedFrameProduct(
        frame_id=frame.frame_id,
        timestamp_ns=frame.timestamp_ns,
        width=frame.width,
        height=frame.height,
        luma_proxy=np.full(shape, 0.2, dtype=np.float32),
        reliability_mask=np.full(shape, reliability, dtype=np.float32),
        radiometric_uncertainty=np.full(shape, 1.0 - reliability, dtype=np.float32),
        saturation_mask=np.zeros(shape, dtype=bool),
        bloom_mask=np.zeros(shape, dtype=bool),
        glare_mask=np.zeros(shape, dtype=bool),
        exposure_factor=1.25,
        reliability_score=reliability,
        quality_flags=("synthetic",),
    )


def make_sequence(
    value: float,
    *,
    reliability: float = 0.9,
    detector_score: float = 0.88,
    track_confidence: float = 0.88,
    saturation_fraction: float = 0.0,
    lost_count_norm: float = 0.0,
    valid_frames: int = 16,
) -> LampCropSequence:
    crops = np.zeros((16, 64, 64, 3), dtype=np.float32)
    crops[:valid_frames] = value
    valid = np.zeros((16,), dtype=bool)
    valid[:valid_frames] = True
    tokens = np.zeros((16, len(TOKEN_NAMES)), dtype=np.float32)
    tokens[:, 0] = 1.0
    tokens[:, 1] = saturation_fraction
    tokens[:, 4] = reliability
    tokens[:, 5] = detector_score
    tokens[:, 6] = track_confidence
    tokens[:, 7] = lost_count_norm
    tokens[:, 8] = 0.8
    return LampCropSequence(
        track_id="lamp_1",
        crop_tensor=crops,
        valid_mask=valid,
        frame_ids=tuple(range(16)),
        timestamps_ns=tuple(range(16)),
        bbox_xyxy=np.zeros((16, 4), dtype=np.float32),
        metadata_tokens=tokens,
        token_names=TOKEN_NAMES,
    )


def make_status_training_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    clip_dir = dataset / "clips"
    frame_dir = clip_dir / "frames"
    annotations = dataset / "annotations"
    frame_dir.mkdir(parents=True)
    annotations.mkdir(parents=True)

    image = Image.new("RGB", (80, 60), (18, 18, 20))
    for x in range(32, 39):
        for y in range(22, 30):
            image.putpixel((x, y), (245, 245, 230))
    image.save(frame_dir / "000001.jpg")

    clip = {
        "clip_id": "clip_status",
        "device_id": "phone_test",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "frames": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "image_uri": "frames/000001.jpg",
                "image_format": "jpg",
                "width": 80,
                "height": 60,
                "camera": {"exposure_time_s": 0.0167, "sensor_sensitivity_iso": 800, "metadata_quality": "partial"},
                "pose": {"imu_quality": "missing"},
            }
        ],
        "tracks": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "track_id": "lamp_1",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [30, 20, 42, 34],
                "bbox_format": "pixel_xyxy_original_frame",
                "detector_score": 0.9,
                "track_confidence": 0.9,
            }
        ],
        "optional_calibration": {"photometric": {}, "map_priors": {}},
    }
    (clip_dir / "clip_status.json").write_text(json.dumps(clip, indent=2), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"dataset_type": "rbccps_measurement", "clips": [{"clip_id": "clip_status", "manifest": "clips/clip_status.json"}]}),
        encoding="utf-8",
    )
    (annotations / "lamp_status.csv").write_text("clip_id,track_id,status\nclip_status,lamp_1,on\n", encoding="utf-8")
    return dataset


def test_crop_sequence_extracts_normalized_bbox_and_pads_mask(tmp_path: Path):
    frame_root = tmp_path / "clip"
    (frame_root / "frames").mkdir(parents=True)
    Image.new("RGB", (80, 60), (40, 40, 40)).save(frame_root / "frames" / "000001.jpg")
    frame = make_frame()
    track = make_track((0.25, 0.25, 0.5, 0.5), "normalized_xyxy_original_frame")

    sequence = build_lamp_crop_sequence(
        "lamp_1",
        [track],
        {1: frame},
        frame_root,
        normalized_products={1: make_product(frame)},
        config=CropSequenceConfig(sequence_length=16, crop_size=64),
    )

    assert sequence.crop_tensor.shape == (16, 64, 64, 3)
    assert sequence.valid_mask.tolist().count(True) == 1
    assert sequence.valid_mask[-1]
    assert sequence.bbox_xyxy[-1, 0] < 20
    assert sequence.bbox_xyxy[-1, 2] > 40
    assert sequence.metadata_tokens[-1, TOKEN_NAMES.index("exposure_factor")] == 1.25
    assert "sequence_padded" in sequence.quality_flags


def test_missing_image_never_produces_off(tmp_path: Path):
    frame = make_frame()
    track = make_track()
    sequence = build_lamp_crop_sequence("lamp_1", [track], {1: frame}, tmp_path)

    output = deterministic_latent_status(sequence)

    assert "image_missing" in sequence.quality_flags
    assert output.status_label == "unknown"
    assert output.status_label != "off"


def test_deterministic_status_bright_dim_dark_and_saturated_semantics():
    assert deterministic_latent_status(make_sequence(0.95)).status_label == "on"
    assert deterministic_latent_status(make_sequence(0.25)).status_label == "dim"
    assert deterministic_latent_status(make_sequence(0.02, detector_score=0.8, reliability=0.8)).status_label == "off"

    saturated = deterministic_latent_status(make_sequence(0.95, reliability=0.35, saturation_fraction=0.12))
    assert saturated.status_label == "saturated"
    assert saturated.saturated_flag
    assert "status_saturation_unsafe" in saturated.quality_flags


def test_temporal_variation_and_lost_count_raise_flicker_index():
    sequence = make_sequence(0.2, lost_count_norm=0.9)
    sequence.crop_tensor[::2] = 0.95

    output = deterministic_latent_status(sequence)

    assert output.flicker_index > 0.55
    assert output.status_label == "flicker"


def test_status_train_module_writes_checkpoint_and_loads_estimator(tmp_path: Path, monkeypatch):
    dataset = make_status_training_dataset(tmp_path)
    out = tmp_path / "status_train"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train-module",
            "--module",
            "status",
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--skip-readiness",
        ],
    )

    train_module_main()

    plan = json.loads((out / "training_plan.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((out / "status_checkpoint.json").read_text(encoding="utf-8"))
    estimator = StatusEstimator.from_checkpoint(out / "status_checkpoint.json")

    assert plan["checkpoint"] == "status_checkpoint.json"
    assert checkpoint["module"] == "status"
    assert checkpoint["crop_config"]["sequence_length"] == 16
    assert checkpoint["model_config"]["crop_size"] == 64
    assert checkpoint["fallback"] == "deterministic_latent_status_v1"
    assert estimator.predict(make_sequence(0.95)).status_label in checkpoint["label_maps"]["status"]


def test_status_train_module_dry_run_only_writes_plan(tmp_path: Path, monkeypatch):
    dataset = make_status_training_dataset(tmp_path)
    out = tmp_path / "status_dry_run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train-module",
            "--module",
            "status",
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--skip-readiness",
            "--dry-run",
        ],
    )

    train_module_main()

    assert (out / "training_plan.json").exists()
    assert not (out / "status_checkpoint.json").exists()
