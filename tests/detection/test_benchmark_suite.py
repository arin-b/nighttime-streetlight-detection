from argparse import Namespace
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
    assert diagnostics["expected_pipeline_runs"] == 16
    assert diagnostics["expected_deployment_runs"] == 80
    assert diagnostics["expected_total_records"] == 105
    assert set(DEPLOYMENT_FORMATS).issuperset(args.deployment_formats)


def test_run_suite_dry_run_writes_manifest_first(tmp_path: Path):
    args = parse_args(["--dry-run", "--output-root", str(tmp_path), "--stage", "detection"])

    assert run_suite(args) == 0
    manifest = tmp_path / "manifests" / "run_manifest.json"
    diagnostics = tmp_path / "manifests" / "dry_run_diagnostics.json"

    assert manifest.exists()
    assert diagnostics.exists()
    assert "D1" in manifest.read_text(encoding="utf-8")


def test_factorial_effects_computes_main_effects():
    rows = [
        {"experiment_id": det.experiment_id, "case": det.case, "map50": idx / 10}
        for idx, det in enumerate(DETECTOR_GRID, start=1)
    ]

    effects = factorial_effects(rows)

    assert effects["available"] is True
    assert set(effects["main_effects"]) == {"cse", "negative_attention", "geometry_attention"}


def test_shortlist_uses_measured_accuracy_efficiency_score():
    rows = [
        {"experiment_id": "D1", "case": "baseline", "f1": 0.7, "map50": 0.8, "map50_95": 0.5, "fps": 30, "model_size_mb": 50},
        {"experiment_id": "D2", "case": "cse", "f1": 0.8, "map50": 0.85, "map50_95": 0.55, "fps": 20, "model_size_mb": 60},
        {"experiment_id": "D3", "case": "negative", "f1": 0.6, "map50": 0.7, "map50_95": 0.4, "fps": 80, "model_size_mb": 40},
    ]

    shortlist = select_deployment_shortlist(rows, limit=2)

    assert [item["experiment_id"] for item in shortlist] == ["D2", "D1"]
    assert "selection_rule" in shortlist[0]


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
