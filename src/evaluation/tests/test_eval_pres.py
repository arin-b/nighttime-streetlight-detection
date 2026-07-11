from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from eval_pres.detection_tracking_artifacts import build_detection_tracking_artifacts
from eval_pres.doctor import run_doctor
from eval_pres.io import build_inputs
from eval_pres.metrics import evaluate
from eval_pres.measurement_artifacts import build_measurement_artifacts
from eval_pres.video_demo import FrameDetection, apply_preset, build_clip_manifest, link_detections


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    Image.new("RGB", (640, 480), (20, 20, 20)).save(frames_dir / "000001.jpg")
    Image.new("RGB", (640, 480), (25, 25, 25)).save(frames_dir / "000002.jpg")
    manifest = {
        "clip_id": "clip_eval",
        "device_id": "phone_a",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "frames": [
            {"frame_id": 1, "timestamp_ns": 1, "image_uri": "frames/000001.jpg", "image_format": "jpg", "width": 640, "height": 480},
            {"frame_id": 2, "timestamp_ns": 2, "image_uri": "frames/000002.jpg", "image_format": "jpg", "width": 640, "height": 480},
        ],
        "tracks": [
            {
                "frame_id": 1,
                "timestamp_ns": 1,
                "track_id": "pred_a",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [100, 50, 140, 110],
                "detector_score": 0.95,
            },
            {
                "frame_id": 2,
                "timestamp_ns": 2,
                "track_id": "pred_a",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [101, 51, 141, 111],
                "detector_score": 0.93,
            },
            {
                "frame_id": 1,
                "timestamp_ns": 1,
                "track_id": "fp_1",
                "class_name": "streetlight_lamp_head",
                "bbox_xyxy": [300, 300, 340, 360],
                "detector_score": 0.40,
            },
        ],
    }
    gt = {
        "metadata": {"route_distance_km": 2.0},
        "lamps": [
            {
                "physical_lamp_id": "lamp_1",
                "inventory_id": "inv_1",
                "frame_id": 1,
                "bbox_xyxy": [100, 50, 140, 110],
                "status": "on",
                "illumination_class": "poor",
                "affected_region_polygon": [[80, 160], [220, 160], [220, 260], [80, 260]],
                "served_area_fraction": 0.5,
                "confounder_present": True,
                "target_attribution_correct": False,
            },
            {
                "physical_lamp_id": "lamp_1",
                "inventory_id": "inv_1",
                "frame_id": 2,
                "bbox_xyxy": [101, 51, 141, 111],
                "status": "on",
                "illumination_class": "poor",
                "affected_region_polygon": [[80, 160], [220, 160], [220, 260], [80, 260]],
                "served_area_fraction": 0.5,
                "confounder_present": True,
                "target_attribution_correct": False,
            },
        ],
    }
    reports = [
        {
            "lamp_observation_id": "obs_1",
            "lamp_track_id": "pred_a",
            "mapped_lamp_id": "inv_1",
            "clip_id": "clip_eval",
            "status": {"label": "on"},
            "affected_region": {
                "area_fraction": 0.55,
                "polygon": [[80, 160], [220, 160], [220, 260], [80, 260]],
            },
            "metrics": {
                "overall_category": "adequate",
                "overall_useful_illumination_score": 0.8,
                "attribution_score": 0.9,
            },
            "confidence": {"overall": 0.9},
            "uncertainty_flags": [],
            "optional_physical_estimates": {"valid": False},
        }
    ]
    manifest_path = tmp_path / "clip_manifest.json"
    gt_path = tmp_path / "ground_truth.json"
    reports_path = tmp_path / "reports.json"
    write_json(manifest_path, manifest)
    write_json(gt_path, gt)
    write_json(reports_path, reports)
    return manifest_path, reports_path, gt_path


def test_metrics_compute_from_synthetic_fixture(tmp_path: Path) -> None:
    manifest, reports, gt = make_fixture(tmp_path)
    inputs = build_inputs(manifest, reports, gt, route_distance_km=None, latency_seconds=1.2, model_paths=[])
    metrics, _ = evaluate(inputs)
    values = {metric.name: metric.value for metric in metrics if metric.status == "computed"}
    assert values["Detector Recall"] == 1.0
    assert abs(values["Detector Precision"] - (2 / 3)) < 1e-5
    assert values["Inventory Match Accuracy"] == 1.0
    assert values["Poor-As-Adequate Illumination Error Rate"] == 1.0
    assert values["Spatial Coverage Bias"] == 0.05


def test_cli_writes_summary_and_plots(tmp_path: Path) -> None:
    manifest, reports, gt = make_fixture(tmp_path)
    out = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval_pres.cli",
            "--manifest",
            str(manifest),
            "--reports",
            str(reports),
            "--ground-truth",
            str(gt),
            "--latency-seconds",
            "1.2",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
    assert (out / "evaluation_summary.json").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "metric_status.csv").exists()
    assert (out / "plots" / "metric_scorecard.png").exists()
    summary = json.loads((out / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["matched_detections_iou_050"] == 2
    assert (out / "command.json").exists()
    assert (out / "environment.json").exists()
    assert (out / "run_summary.json").exists()


def test_unified_cli_help_and_evaluate_alias(tmp_path: Path) -> None:
    manifest, reports, gt = make_fixture(tmp_path)
    help_result = subprocess.run(
        [sys.executable, "-m", "eval_pres", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert help_result.returncode == 0, help_result.stdout
    assert "demo-video" in help_result.stdout
    assert "doctor" in help_result.stdout

    out = tmp_path / "unified_out"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval_pres",
            "evaluate",
            "--manifest",
            str(manifest),
            "--reports",
            str(reports),
            "--ground-truth",
            str(gt),
            "--latency-seconds",
            "1.2",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
    assert (out / "evaluation_summary.json").exists()
    assert (out / "logs" / "run.log").exists()


def test_detection_and_measurement_artifacts_are_written(tmp_path: Path) -> None:
    manifest, reports, gt = make_fixture(tmp_path)
    det_out = tmp_path / "det_artifacts"
    meas_out = tmp_path / "measurement_artifacts"
    det_artifacts = build_detection_tracking_artifacts(
        manifest_path=manifest,
        out_dir=det_out,
        frame_root=tmp_path,
        ground_truth_path=gt,
        render_overlays=True,
        video_fps=2.0,
    )
    meas_artifacts = build_measurement_artifacts(
        reports_path=reports,
        out_dir=meas_out,
        manifest_path=manifest,
        frame_root=tmp_path,
    )
    assert (det_out / "detection_summary_table.csv").exists()
    assert (det_out / "tracking_overlay_video.mp4").exists()
    assert (det_out / "detection_tracking_plots" / "detections_per_frame.png").exists()
    assert (meas_out / "measurement_summary_table.csv").exists()
    assert (meas_out / "measurement_summary_table.html").exists()
    assert (meas_out / "measurement_plots" / "measurement_score_distribution.png").exists()
    assert (det_out / "track_summary.csv").exists()
    assert (det_out / "tracking_events.csv").exists()
    assert det_artifacts["track_cards"]
    assert meas_artifacts["per_lamp_cards"]


def test_video_demo_linker_and_manifest_builder(tmp_path: Path) -> None:
    frames = []
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(1, 3):
        path = frame_dir / f"{index:06d}.jpg"
        Image.new("RGB", (120, 80), (10, 10, 10)).save(path)
        frames.append(path)
    detections = [
        [FrameDetection(1, (10, 10, 30, 30), 0.9)],
        [FrameDetection(2, (11, 10, 31, 30), 0.85)],
    ]
    linked = link_detections(detections, iou_threshold=0.3)
    manifest = build_clip_manifest(frames, linked, fps_sample=2.0, clip_id="tiny")
    assert len(linked) == 1
    assert manifest["tracks"][0]["track_id"] == "eval_lamp_0001"
    assert len(manifest["frames"]) == 2


def test_video_demo_preset_defaults() -> None:
    import argparse

    args = argparse.Namespace(preset="quick", fps_sample=3.0, max_det=12, measurement_max_tracks=30)
    apply_preset(args)
    assert args.fps_sample == 1.0
    assert args.max_det == 8
    assert args.measurement_max_tracks == 15

    args = argparse.Namespace(preset="full", fps_sample=3.0, max_det=12, measurement_max_tracks=30)
    apply_preset(args)
    assert args.fps_sample == 3.0
    assert args.max_det == 20
    assert args.measurement_max_tracks == 60


def test_doctor_report_can_be_mocked(monkeypatch, tmp_path: Path) -> None:
    from eval_pres import doctor

    monkeypatch.setattr(doctor, "_check_python", lambda: {"name": "python_version", "status": "pass"})
    monkeypatch.setattr(doctor, "_check_import", lambda *args, **kwargs: {"name": args[0], "status": "pass"})
    monkeypatch.setattr(doctor, "_check_executable", lambda *args, **kwargs: {"name": args[0], "status": "warn"})
    monkeypatch.setattr(doctor, "_check_path", lambda *args, **kwargs: {"name": args[0], "status": "pass"})
    monkeypatch.setattr(doctor, "_check_measurement_import", lambda root: {"name": "rbccps_measurement", "status": "pass"})
    monkeypatch.setattr(doctor, "_check_write_access", lambda root: {"name": "write_access", "status": "pass"})
    report = run_doctor(tmp_path)
    assert report["status"] == "warn"
    assert (tmp_path / "doctor_report.json").exists()
