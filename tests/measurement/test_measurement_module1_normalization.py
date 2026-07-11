import json
from pathlib import Path

import numpy as np
from PIL import Image

from rbccps_measurement.contracts.input_schema import CameraMetadata, FrameRecord, PoseRecord
from rbccps_measurement.cli.train_module import main as train_module_main
from rbccps_measurement.normalization.module1 import CaptureNormalizer
from rbccps_measurement.training.normalization import train_capture_normalization


def make_frame(width=32, height=24) -> FrameRecord:
    return FrameRecord(
        frame_id=1,
        timestamp_ns=1000,
        image_uri="frames/000001.jpg",
        image_format="jpg",
        width=width,
        height=height,
        camera=CameraMetadata(
            exposure_time_s=0.0167,
            sensor_sensitivity_iso=800,
            ae_mode="auto",
            hdr_mode="unknown",
            night_mode=True,
            metadata_quality="partial",
        ),
        pose=PoseRecord(imu_quality="missing"),
    )


def make_training_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    clip_dir = dataset / "clips"
    frame_dir = clip_dir / "frames"
    frame_dir.mkdir(parents=True)
    image = Image.new("RGB", (32, 24), (20, 20, 24))
    image.putpixel((12, 4), (255, 255, 255))
    image.save(frame_dir / "000001.jpg")

    clip_payload = {
        "clip_id": "clip_train",
        "device_id": "phone_test",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "frames": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "image_uri": "frames/000001.jpg",
                "image_format": "jpg",
                "width": 32,
                "height": 24,
                "camera": {
                    "exposure_time_s": 0.02,
                    "sensor_sensitivity_iso": 1000,
                    "ae_mode": "auto",
                    "metadata_quality": "partial",
                },
                "pose": {"imu_quality": "missing"},
            }
        ],
        "tracks": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "track_id": "track_1",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [10, 2, 16, 8],
                "bbox_format": "pixel_xyxy_original_frame",
                "detector_score": 0.8,
            }
        ],
        "optional_calibration": {"photometric": {"field_lux_calibration_id": None}, "map_priors": {}},
    }
    (clip_dir / "clip_train.json").write_text(json.dumps(clip_payload, indent=2), encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_type": "rbccps_measurement",
                "clips": [{"clip_id": "clip_train", "manifest": "clips/clip_train.json"}],
            }
        ),
        encoding="utf-8",
    )
    return dataset


def test_capture_normalizer_emits_locked_module1_artifact():
    rgb = np.zeros((24, 32, 3), dtype=np.float32)
    rgb[:, :, :] = 0.08
    rgb[2:6, 10:14, :] = 1.0

    product = CaptureNormalizer().normalize_array(rgb, make_frame())
    summary = product.summary()

    assert product.luma_proxy.shape == (24, 32)
    assert product.reliability_mask.shape == (24, 32)
    assert product.radiometric_uncertainty.shape == (24, 32)
    assert product.saturation_mask.any()
    assert product.bloom_mask.any()
    assert "saturation" in product.quality_flags
    assert "auto_exposure_active" in product.quality_flags
    assert 0.0 < summary["reliability_score"] <= 1.0
    assert summary["exposure_factor"] == 1.0


def test_normalization_training_writes_loadable_checkpoint(tmp_path: Path):
    dataset = make_training_dataset(tmp_path)
    frame_dir = dataset / "clips" / "frames"

    result = train_capture_normalization(dataset, tmp_path / "train_out")
    assert result.frames_seen == 1
    assert result.frames_used == 1
    assert result.checkpoint_path.exists()

    normalizer = CaptureNormalizer.from_checkpoint(result.checkpoint_path)
    product = normalizer.normalize_path(frame_dir / "000001.jpg", make_frame())
    assert product.metadata["checkpoint_path"] == str(result.checkpoint_path)


def test_train_module_cli_trains_normalization_checkpoint(tmp_path: Path, monkeypatch):
    dataset = make_training_dataset(tmp_path)
    out = tmp_path / "cli_train"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train-module",
            "--module",
            "normalization",
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--skip-readiness",
        ],
    )

    train_module_main()

    plan = json.loads((out / "training_plan.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((out / "normalization_checkpoint.json").read_text(encoding="utf-8"))
    assert plan["status"] == "trained"
    assert plan["checkpoint"] == "normalization_checkpoint.json"
    assert checkpoint["checkpoint_type"] == "capture_normalization_checkpoint"
