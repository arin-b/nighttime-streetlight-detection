from argparse import Namespace
import json
import os
from types import SimpleNamespace
from pathlib import Path

from rbccps_od.training.benchmark_suite import (
    DEPLOYMENT_FORMATS,
    DETECTOR_GRID,
    aggregate_results,
    dry_run_diagnostics,
    factorial_effects,
    parse_args,
    pipeline_grid,
    planned_records,
    run_suite,
    select_deployment_shortlist,
)


def test_detector_grid_uses_required_d_ids_and_factorial_order():
    assert [det.experiment_id for det in DETECTOR_GRID] == [f"D{idx}" for idx in range(1, 9)]
    assert [det.case for det in DETECTOR_GRID] == [
        "baseline",
        "cse",
        "negative",
        "geometry",
        "negative_cse",
        "geometry_cse",
        "negative_geometry",
        "all_modules",
    ]
    assert DETECTOR_GRID[-1].use_cse is True
    assert DETECTOR_GRID[-1].use_negative_attention is True
    assert DETECTOR_GRID[-1].use_geometry_attention is True


def test_pipeline_grid_maps_p1_p16_to_filtering_states():
    grid = pipeline_grid()
    assert [item.experiment_id for item in grid[:8]] == [f"P{idx}" for idx in range(1, 9)]
    assert [item.experiment_id for item in grid[8:]] == [f"P{idx}" for idx in range(9, 17)]
    assert all(not item.filtering_enabled for item in grid[:8])
    assert all(item.filtering_enabled for item in grid[8:])
    assert [item.detector_id for item in grid[:8]] == [f"D{idx}" for idx in range(1, 9)]
    assert [item.detector_id for item in grid[8:]] == [f"D{idx}" for idx in range(1, 9)]


def test_planned_dry_run_graph_counts(tmp_path: Path):
    args = parse_args(["--dry-run", "--output-root", str(tmp_path)])
    records = planned_records(args)
    diagnostics = dry_run_diagnostics(args, records)

    assert diagnostics["expected_detector_runs"] == 8
    assert diagnostics["expected_detector_eval_runs"] == 8
    assert diagnostics["expected_pipeline_runs"] == 16
    assert diagnostics["expected_deployment_runs"] == 40
    assert diagnostics["expected_total_records"] == 73
    assert set(DEPLOYMENT_FORMATS).issuperset(args.deployment_formats)


def test_run_suite_dry_run_writes_manifest_first(tmp_path: Path):
    args = parse_args(["--dry-run", "--output-root", str(tmp_path), "--stage", "detection"])

    assert run_suite(args) == 0
    manifest = tmp_path / "manifests" / "run_manifest.json"
    diagnostics = tmp_path / "manifests" / "dry_run_diagnostics.json"

    assert manifest.exists()
    assert diagnostics.exists()
    assert "D1" in manifest.read_text(encoding="utf-8")
    assert "detector_eval" in manifest.read_text(encoding="utf-8")


def test_cli_separates_detector_eval_data_from_pipeline_video(tmp_path: Path):
    data = tmp_path / "dataset.yaml"
    video = tmp_path / "clip.mp4"
    data.write_text("path: .\ntrain: images/train\nval: images/valid\ntest: images/test\n", encoding="utf-8")
    video.write_bytes(b"video")

    args = parse_args(
        [
            "--dry-run",
            "--output-root",
            str(tmp_path / "runs"),
            "--detector-eval-data",
            str(data),
            "--pipeline-video",
            str(video),
        ]
    )
    records = planned_records(args)
    detector_eval = next(record for record in records if record.stage == "detector_eval")
    filtering = next(record for record in records if record.stage == "filtering")

    assert str(data) in detector_eval.command
    assert str(video) in filtering.command


def test_calibration_dataset_yaml_is_passed_to_export_data(tmp_path: Path):
    data = tmp_path / "calibration_dataset.yaml"
    data.write_text("path: .\ntrain: images/train\nval: images/valid\n", encoding="utf-8")
    args = parse_args(
        [
            "--stage",
            "deployment",
            "--output-root",
            str(tmp_path / "runs"),
            "--calibration-dataset-yaml",
            str(data),
            "--shortlist-size",
            "1",
        ]
    )

    record = planned_records(args)[0]

    assert "--data" in record.command
    assert str(data) in record.command


def test_legacy_calibration_data_alias_still_maps_to_dataset_yaml(tmp_path: Path):
    data = tmp_path / "calibration_dataset.yaml"
    data.write_text("path: .\ntrain: images/train\nval: images/valid\n", encoding="utf-8")

    args = parse_args(["--calibration-data", str(data)])

    assert args.calibration_dataset_yaml == str(data)


def test_factorial_effects_computes_main_effects():
    rows = [
        {"experiment_id": det.experiment_id, "case": det.case, "map50_95": idx / 10}
        for idx, det in enumerate(DETECTOR_GRID, start=1)
    ]

    effects = factorial_effects(rows)

    assert effects["available"] is True
    assert set(effects["main_effects"]) == {"cse", "negative_attention", "geometry_attention"}
    assert len(effects["interaction_effects"]) == 4


def test_shortlist_uses_pareto_efficient_pipeline_metrics():
    rows = [
        {"experiment_id": "P1", "detector_id": "D1", "filtering_enabled": False,
         "map50_95": 0.50, "precision": 0.7, "recall": 0.7, "fps": 30, "model_size_mb": 50},
        {"experiment_id": "P2", "detector_id": "D2", "filtering_enabled": False,
         "map50_95": 0.55, "precision": 0.8, "recall": 0.8, "fps": 20, "model_size_mb": 60},
        {"experiment_id": "P3", "detector_id": "D3", "filtering_enabled": False,
         "map50_95": 0.40, "precision": 0.6, "recall": 0.6, "fps": 15, "model_size_mb": 70},
    ]

    shortlist = select_deployment_shortlist(rows, limit=2)

    assert [item["experiment_id"] for item in shortlist] == ["P2", "P1"]
    assert all("Pareto" in item["selection_rule"] for item in shortlist)
    assert select_deployment_shortlist(rows, limit=2, min_precision=0.75) == [shortlist[0]]


def test_deployment_record_has_one_export_command_and_a_matching_audit_command(tmp_path: Path):
    from rbccps_od.training.benchmark_suite import _deployment_audit_command
    args = parse_args([
        "--stage", "deployment", "--output-root", str(tmp_path),
        "--pipeline-video", str(tmp_path / "clip.mp4"), "--gt-labels", str(tmp_path / "labels"),
        "--shortlist-size", "1",
    ])
    record = planned_records(args)[0]

    assert record.experiment_id == "P1_TFLITE_FP32"
    assert record.metadata["pipeline_id"] == "P1"
    assert record.metadata["format"] == "tflite_fp32"
    assert "&&" not in record.command
    command = _deployment_audit_command(args, record, tmp_path / "best.tflite")
    assert "--disable-multicue" in command
    assert "--gt-labels" in command
    assert "--device" in command
    assert "cpu" in command



def test_aggregate_results_reads_existing_audit_report(tmp_path: Path):
    args = parse_args(["--output-root", str(tmp_path), "--stage", "report"])
    p1 = tmp_path / "pipeline" / "P1"
    p1.mkdir(parents=True)
    (p1 / "audit_report.json").write_text(
        """
{
  "video": {"frame_count": 10},
  "evaluation_metrics": {
    "detection": {"AP@0.5": 0.75, "mAP@0.5:0.95": 0.45},
    "frame_detection": {"precision": 0.8, "recall": 0.6, "f1": 0.6857, "fp": 2},
    "evaluation_context": {"frames_evaluated": 10, "raw_target_prediction_instances": 7, "filtered_target_prediction_instances": 6}
  }
}
""".strip(),
        encoding="utf-8",
    )

    summary = aggregate_results(args)

    assert summary["pipelines"][0]["map50"] == 0.75
    assert summary["pipelines"][0]["false_positives_per_frame"] == 0.2
    assert (tmp_path / "aggregated" / "pipeline_results.csv").exists()
    assert (tmp_path / "reports" / "final_report.md").exists()



def test_deployment_execution_reuses_export_and_writes_heldout_metrics(tmp_path: Path, monkeypatch):
    import rbccps_od.training.benchmark_suite as suite

    args = parse_args([
        "--stage", "deployment", "--output-root", str(tmp_path),
        "--pipeline-video", str(tmp_path / "clip.mp4"),
        "--gt-labels", str(tmp_path / "labels"),
        "--shortlist-size", "1", "--artifact-root", str(tmp_path / "artifacts"),
    ])
    record = planned_records(args)[0]
    checkpoint = Path(record.metadata["checkpoint"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    exported = Path(record.metadata["export_dir"]) / "tflite_fp32" / "model.tflite"
    exported.parent.mkdir(parents=True)
    exported.write_bytes(b"exported")
    os.utime(exported, (checkpoint.stat().st_mtime + 1, checkpoint.stat().st_mtime + 1))
    (Path(record.metadata["export_dir"]) / "export_report.json").write_text(json.dumps({
        "results": [{"format_name": "tflite_fp32", "status": "success", "output_path": str(exported)}],
    }), encoding="utf-8")

    def fake_run(command, **_kwargs):
        assert "evaluation.eval_pres.run_audit" in command
        audit_dir = Path(record.artifact_dir)
        audit_dir.mkdir(parents=True)
        (audit_dir / "audit_report.json").write_text(json.dumps({
            "video": {"frame_count": 3},
            "evaluation_metrics": {
                "detection": {"AP@0.5": 0.5, "mAP@0.5:0.95": 0.3},
                "frame_detection": {"precision": 0.5, "recall": 0.6, "f1": 0.545, "fp": 1},
                "evaluation_context": {"frames_evaluated": 3},
            },
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", fake_run)
    log = tmp_path / "logs" / "deployment.log"
    log.parent.mkdir(parents=True)
    result = suite._execute_deployment_record(args, record, log)

    assert result.status == "success"
    assert result.metadata["exported_model"] == str(exported)
    timing = json.loads((Path(record.artifact_dir) / "benchmark_timing.json").read_text())
    assert timing["frames_evaluated"] == 3
    assert timing["model_size_mb"] is not None


def test_stage_provenance_allows_deployment_expansion_and_ignores_report(tmp_path: Path):
    from rbccps_od.training.benchmark_suite import _run_signature, _stage_signature_mismatches
    shortlist = parse_args(["--output-root", str(tmp_path), "--deployment-mode", "shortlist"])
    expanded = parse_args(["--output-root", str(tmp_path), "--deployment-mode", "all", "--resume"])
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "deployment_P1_TFLITE_FP32.json").write_text(json.dumps({
        "stage": "deployment", "experiment_id": "P1_TFLITE_FP32",
        "status": "success", "run_signature": _run_signature(shortlist),
    }))
    (manifest_dir / "report_REPORT.json").write_text(json.dumps({
        "stage": "report", "experiment_id": "REPORT", "status": "success",
        "run_signature": {"imgsz": 999},
    }))
    assert _stage_signature_mismatches(expanded, manifest_dir) == []
