import pytest

from rbccps_measurement.contracts.calibration_policy import CalibrationPolicy
from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.ingest.validation import validate_clip_manifest


def make_manifest_payload(**overrides):
    payload = {
        "clip_id": "clip_001",
        "device_id": "pixel_test",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "frames": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "image_uri": "frames/000001.jpg",
                "image_format": "RGB",
                "width": 1920,
                "height": 1080,
                "camera": {
                    "exposure_time_s": 0.0167,
                    "sensor_sensitivity_iso": 800,
                    "ae_mode": "auto",
                    "hdr_mode": "off",
                    "night_mode": False,
                    "metadata_quality": "partial",
                },
                "pose": {
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "gps_accuracy_m": 4.5,
                    "heading_deg": 82.0,
                    "imu_quality": "good",
                },
            }
        ],
        "tracks": [
            {
                "frame_id": 1,
                "timestamp_ns": 1000,
                "track_id": "track_1",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [800, 120, 850, 180],
                "bbox_format": "pixel_xyxy_original_frame",
                "detector_score": 0.82,
                "track_confidence": 0.74,
                "track_age": 3,
                "lost_count": 0,
                "source_model": "test_detector",
            }
        ],
        "optional_calibration": {
            "photometric": {"field_lux_calibration_id": None},
            "map_priors": {},
        },
    }
    payload.update(overrides)
    return payload


def test_clip_manifest_validates_basic_contract():
    manifest = ClipManifest.from_dict(make_manifest_payload())
    validate_clip_manifest(manifest)
    assert manifest.clip_id == "clip_001"
    assert manifest.tracks[0].bbox_format == "pixel_xyxy_original_frame"
    assert manifest.tracks[0].class_name == "streetlight_lamp_head"


def test_clip_manifest_normalizes_legacy_streetlight_class():
    payload = make_manifest_payload()
    payload["tracks"][0]["class_name"] = "streetlight"
    manifest = ClipManifest.from_dict(payload)
    validate_clip_manifest(manifest)
    assert manifest.tracks[0].class_name == "streetlight_lamp_head"


def test_clip_manifest_accepts_pole_support_tracks():
    payload = make_manifest_payload()
    pole_track = dict(payload["tracks"][0])
    pole_track["track_id"] = "pole_1"
    pole_track["class_name"] = "streetlight_pole"
    pole_track["bbox_xyxy"] = [790, 120, 860, 500]
    payload["tracks"].append(pole_track)
    manifest = ClipManifest.from_dict(payload)
    validate_clip_manifest(manifest)
    assert {track.class_name for track in manifest.tracks} == {"streetlight_lamp_head", "streetlight_pole"}


def test_validation_rejects_bbox_outside_original_frame():
    payload = make_manifest_payload()
    payload["tracks"][0]["bbox_xyxy"] = [800, 120, 3000, 180]
    manifest = ClipManifest.from_dict(payload)
    with pytest.raises(ValueError, match="outside original frame"):
        validate_clip_manifest(manifest)


def test_validation_rejects_timestamp_mismatch():
    payload = make_manifest_payload()
    payload["tracks"][0]["timestamp_ns"] = 2000
    manifest = ClipManifest.from_dict(payload)
    with pytest.raises(ValueError, match="timestamp does not match"):
        validate_clip_manifest(manifest)


def test_validation_rejects_duplicate_track_frame():
    payload = make_manifest_payload()
    payload["tracks"].append(dict(payload["tracks"][0]))
    manifest = ClipManifest.from_dict(payload)
    with pytest.raises(ValueError, match="duplicate detection"):
        validate_clip_manifest(manifest)


def test_calibration_policy_blocks_physical_estimates_for_low_tier():
    decision = CalibrationPolicy.decide(
        calibration_level=1,
        has_field_lux_calibration=True,
        auto_exposure_active=False,
        metadata_quality="good",
    )
    assert decision.proxy_allowed is True
    assert decision.physical_allowed is False
    assert "below" in decision.physical_reason


def test_calibration_policy_allows_physical_estimates_only_when_controlled():
    decision = CalibrationPolicy.decide(
        calibration_level=3,
        has_field_lux_calibration=True,
        auto_exposure_active=False,
        metadata_quality="good",
    )
    assert decision.physical_allowed is True
