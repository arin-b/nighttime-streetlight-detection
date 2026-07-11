import json
from pathlib import Path

from rbccps_measurement.cli.train_module import main as train_module_main
from rbccps_measurement.contracts.calibration_policy import CalibrationPolicy
from rbccps_measurement.contracts.input_schema import CalibrationRecord
from rbccps_measurement.photometry.sparse_reference_field import estimate_photometric_field, parse_lux_references
from rbccps_measurement.pipeline import run_clip_to_directory

from test_measurement_contracts import make_manifest_payload


def make_calibration(lux_points=None, calibration_id="field_ref_001") -> CalibrationRecord:
    return CalibrationRecord.from_dict(
        {
            "photometric": {
                "field_lux_calibration_id": calibration_id,
                "response_curve_quality": 0.9,
                "vignetting_quality": 0.85,
                "lux_points": lux_points or [],
            },
            "map_priors": {},
        }
    )


def make_policy(level=3, auto_exposure=False):
    return CalibrationPolicy.decide(
        calibration_level=level,
        has_field_lux_calibration=True,
        auto_exposure_active=auto_exposure,
        metadata_quality="good",
    )


def test_low_calibration_level_blocks_physical_output_even_with_references():
    references = parse_lux_references([{"lux_value": 7.5, "point_type": "P1", "orientation": "horizontal"}], clip_id="clip", track_id="track")
    output = estimate_photometric_field(
        track_id="track",
        clip_id="clip",
        calibration=make_calibration(),
        policy=make_policy(level=1),
        useful_score=0.8,
        fusion_confidence=0.9,
        glare_penalty=0.05,
        dark_hole_fraction=0.05,
        confounder_penalty=0.05,
        geometry_quality=0.9,
        metadata_quality="good",
        auto_exposure_active=False,
        references=references,
    )

    assert output.physical_valid is False
    assert "below" in output.reason
    assert output.horizontal_illuminance_lux_mean is None


def test_photometric_quality_is_min_of_signal_and_calibration():
    output = estimate_photometric_field(
        track_id="track",
        clip_id="clip",
        calibration=make_calibration(),
        policy=make_policy(),
        useful_score=0.7,
        fusion_confidence=0.9,
        glare_penalty=0.1,
        dark_hole_fraction=0.1,
        confounder_penalty=0.1,
        geometry_quality=0.85,
        metadata_quality="good",
        auto_exposure_active=False,
    )

    assert output.q_calib == min(output.q_signal, output.q_calibration)
    assert output.physical_valid is True
    assert output.horizontal_illuminance_lux_mean is not None
    assert output.horizontal_illuminance_lux_mean >= 0.0


def test_monotone_proxy_score_and_uncertainty_widens_with_sparse_support():
    dense_rows = [
        {"lux_value": "4.0", "point_type": f"P{idx}", "orientation": "horizontal", "ground_x_m": idx}
        for idx in range(1, 6)
    ]
    dense_refs = parse_lux_references(dense_rows, clip_id="clip", track_id="track")
    sparse_output = estimate_photometric_field(
        track_id="track",
        clip_id="clip",
        calibration=make_calibration(),
        policy=make_policy(),
        useful_score=0.45,
        fusion_confidence=0.85,
        glare_penalty=0.1,
        dark_hole_fraction=0.1,
        confounder_penalty=0.1,
        geometry_quality=0.85,
        metadata_quality="good",
        auto_exposure_active=False,
    )
    low_score = estimate_photometric_field(
        track_id="track",
        clip_id="clip",
        calibration=make_calibration(dense_rows),
        policy=make_policy(),
        useful_score=0.4,
        fusion_confidence=0.85,
        glare_penalty=0.1,
        dark_hole_fraction=0.1,
        confounder_penalty=0.1,
        geometry_quality=0.85,
        metadata_quality="good",
        auto_exposure_active=False,
        references=dense_refs,
    )
    dense_same_score = estimate_photometric_field(
        track_id="track",
        clip_id="clip",
        calibration=make_calibration(dense_rows),
        policy=make_policy(),
        useful_score=0.45,
        fusion_confidence=0.85,
        glare_penalty=0.1,
        dark_hole_fraction=0.1,
        confounder_penalty=0.1,
        geometry_quality=0.85,
        metadata_quality="good",
        auto_exposure_active=False,
        references=dense_refs,
    )
    high_score = estimate_photometric_field(
        track_id="track",
        clip_id="clip",
        calibration=make_calibration(dense_rows),
        policy=make_policy(),
        useful_score=0.75,
        fusion_confidence=0.85,
        glare_penalty=0.1,
        dark_hole_fraction=0.1,
        confounder_penalty=0.1,
        geometry_quality=0.85,
        metadata_quality="good",
        auto_exposure_active=False,
        references=dense_refs,
    )

    sparse_width = sparse_output.horizontal_illuminance_lux_interval[1] - sparse_output.horizontal_illuminance_lux_interval[0]
    dense_width = dense_same_score.horizontal_illuminance_lux_interval[1] - dense_same_score.horizontal_illuminance_lux_interval[0]
    sparse_relative_width = sparse_width / sparse_output.horizontal_illuminance_lux_mean
    dense_relative_width = dense_width / dense_same_score.horizontal_illuminance_lux_mean
    assert high_score.horizontal_illuminance_lux_mean >= low_score.horizontal_illuminance_lux_mean
    assert sparse_relative_width > dense_relative_width
    assert high_score.vertical_illuminance_lux_mean is None


def test_pipeline_photometry_preserves_schema_and_respects_auto_exposure(tmp_path: Path):
    payload = make_manifest_payload(calibration_level=3)
    payload["frames"][0]["camera"]["metadata_quality"] = "good"
    payload["frames"][0]["camera"]["ae_mode"] = "auto"
    payload["optional_calibration"]["photometric"]["field_lux_calibration_id"] = "field_ref_001"
    manifest_path = tmp_path / "clip_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = run_clip_to_directory(manifest_path, tmp_path / "run")[0].to_dict()

    physical = report["optional_physical_estimates"]
    assert physical["valid"] is False
    assert physical["horizontal_illuminance_lux_mean"] is None
    assert "q_signal" in physical
    assert "photometry" in report["traceability"]["model_versions"]


def test_pipeline_photometry_emits_valid_controlled_tier_output(tmp_path: Path):
    payload = make_manifest_payload(calibration_level=3)
    payload["frames"][0]["camera"]["ae_mode"] = "off"
    payload["frames"][0]["camera"]["metadata_quality"] = "good"
    payload["optional_calibration"]["photometric"]["field_lux_calibration_id"] = "field_ref_001"
    manifest_path = tmp_path / "clip_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = run_clip_to_directory(manifest_path, tmp_path / "run")[0].to_dict()
    physical = report["optional_physical_estimates"]

    assert physical["valid"] is True
    assert physical["horizontal_illuminance_lux_mean"] is not None
    assert physical["horizontal_illuminance_lux_interval"][0] <= physical["horizontal_illuminance_lux_mean"] <= physical["horizontal_illuminance_lux_interval"][1]


def test_train_module_photometry_writes_initialized_checkpoint(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset"
    lux = dataset / "lux"
    lux.mkdir(parents=True)
    (lux / "lux_points.csv").write_text("clip_id,frame_id,track_id,point_type,lux_value,x,y,orientation\n", encoding="utf-8")
    out = tmp_path / "train"

    monkeypatch.setattr("sys.argv", ["train-module", "--module", "photometry", "--dataset", str(dataset), "--out", str(out), "--skip-readiness"])
    train_module_main()

    checkpoint = json.loads((out / "photometry_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "initialized_not_optimized"
    assert checkpoint["assumptions"]["sparse_reference_interpretation"] == "sparse_reference_not_dense_ground_truth"
    if checkpoint["weights"]:
        assert (out / checkpoint["weights"]).exists()
