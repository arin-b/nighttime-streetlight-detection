from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rbccps_od.config.paths import repo_root
from rbccps_od.training.ablation import CASE_BY_NAME, EXPERIMENT_BY_NAME, _artifact_name_for


@dataclass(frozen=True)
class DetectorExperiment:
    experiment_id: str
    case: str
    use_cse: bool
    use_negative_attention: bool
    use_geometry_attention: bool


@dataclass(frozen=True)
class PipelineExperiment:
    experiment_id: str
    detector_id: str
    filtering_enabled: bool


@dataclass
class StageRecord:
    experiment_id: str
    stage: str
    status: str = "planned"
    command: list[str] = field(default_factory=list)
    artifact_dir: str = ""
    start_time: str | None = None
    completion_time: str | None = None
    duration_s: float | None = None
    returncode: int | None = None
    error_message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


DETECTOR_GRID: tuple[DetectorExperiment, ...] = (
    DetectorExperiment("D1", "baseline", False, False, False),
    DetectorExperiment("D2", "cse", True, False, False),
    DetectorExperiment("D3", "negative", False, True, False),
    DetectorExperiment("D4", "geometry", False, False, True),
    DetectorExperiment("D5", "negative_cse", True, True, False),
    DetectorExperiment("D6", "geometry_cse", True, False, True),
    DetectorExperiment("D7", "negative_geometry", False, True, True),
    DetectorExperiment("D8", "all_modules", True, True, True),
)

DEPLOYMENT_FORMATS: tuple[str, ...] = ("tflite_fp32", "tflite_fp16", "onnx_fp32", "onnx_fp16", "ncnn")


def pipeline_grid() -> tuple[PipelineExperiment, ...]:
    no_filter = tuple(
        PipelineExperiment(f"P{idx}", detector.experiment_id, False)
        for idx, detector in enumerate(DETECTOR_GRID, start=1)
    )
    filtered = tuple(
        PipelineExperiment(f"P{idx + 8}", detector.experiment_id, True)
        for idx, detector in enumerate(DETECTOR_GRID, start=1)
    )
    return no_filter + filtered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete YOLO26 detector ablation, filtering, and deployment benchmark suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", choices=("all",), default="all", help="Experiment graph to run.")
    parser.add_argument(
        "--stage",
        choices=("all", "detection", "filtering", "deployment", "report"),
        default="all",
        help="Stage to execute.",
    )
    parser.add_argument("--config", help="Optional JSON config file containing CLI argument overrides.")
    parser.add_argument("--output-root", default="runs/ablation_deployment", help="Suite output directory.")
    parser.add_argument("--data-variant", default="original", help="Existing ablation data variant.")
    parser.add_argument("--dry-run", action="store_true", help="Print and save the planned experiment graph only.")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose manifest status is already success.")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun failed stages when used with --resume.")
    parser.add_argument("--max-parallel", type=int, default=1, help="Reserved for future safe parallel execution.")

    parser.add_argument("--model", default=None, help="Pretrained YOLO26m weights passed to the training ablation runner.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--project", default="runs/yolo26m_ablation")
    parser.add_argument("--artifact-root", default="models/fine_tuned/yolo26m_ablation")
    parser.add_argument("--negative-mask-root", default=None)
    parser.add_argument("--negative-mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--allow-dataset-negative-masks", action="store_true")
    parser.add_argument("--original-image-dir", default="datasets/processed/original/images")
    parser.add_argument("--original-label-dir", default=None)
    parser.add_argument("--original-dataset-root", default="datasets/processed/original-yolo26m")
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy", "symlink"), default="auto")
    parser.add_argument("--overwrite-original-view", action="store_true")

    parser.add_argument("--video", help="Held-out video used for detector/filtering evaluation and deployment benchmark.")
    parser.add_argument("--pipeline-video", help="Held-out video used for tracking/filtering/deployment evaluation. Overrides --video.")
    parser.add_argument(
        "--detector-eval-data",
        help=(
            "Dataset YAML for detector-only held-out evaluation. Defaults to the selected data variant's "
            "dataset.yaml when it can be resolved."
        ),
    )
    parser.add_argument("--detector-eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--gt-labels", help="YOLO label directory for held-out evaluation.")
    parser.add_argument("--gt-status-file", help="Optional status-classification ground truth CSV.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--no-save-video", action="store_true")

    parser.add_argument(
        "--calibration-dataset-yaml",
        help=(
            "Dataset YAML/config used to construct the representative calibration set for export formats "
            "that require calibration. The default FP32/FP16/NCNN formats do not require it."
        ),
    )
    parser.add_argument(
        "--calibration-data",
        dest="calibration_dataset_yaml",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--deployment-formats", nargs="*", default=list(DEPLOYMENT_FORMATS), choices=DEPLOYMENT_FORMATS)
    parser.add_argument("--deployment-mode", choices=("shortlist", "all"), default="shortlist")
    parser.add_argument("--shortlist-size", type=int, default=8)
    parser.add_argument("--deployment-preset", choices=("quick", "standard", "full"), default="quick")
    parser.add_argument("--deployment-max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    if args.config:
        args = _merge_config_args(args, Path(args.config))
    return args


def _merge_config_args(args: argparse.Namespace, config_path: Path) -> argparse.Namespace:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("--config JSON must contain an object of argument names to values.")
    for key, value in config.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, value)
    return args


def suite_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = _resolve(args.output_root)
    return {
        "root": root,
        "manifests": root / "manifests",
        "logs": root / "logs",
        "pipeline": root / "pipeline",
        "detector_eval": root / "detector_eval",
        "exports": root / "deployment" / "exports",
        "benchmarks": root / "deployment" / "benchmarks",
        "aggregated": root / "aggregated",
        "plots": root / "plots",
        "reports": root / "reports",
    }


def planned_records(args: argparse.Namespace) -> list[StageRecord]:
    records: list[StageRecord] = []
    detectors = list(DETECTOR_GRID)
    pipelines = list(pipeline_grid())
    if args.stage in {"all", "detection"}:
        for detector in detectors:
            records.append(_detector_record(args, detector))
            records.append(_detector_eval_record(args, detector))
    if args.stage in {"all", "filtering"}:
        for pipeline in pipelines:
            records.append(_pipeline_record(args, pipeline))
    if args.stage in {"all", "deployment"}:
        selected = detectors if args.deployment_mode == "all" else detectors[: min(args.shortlist_size, len(detectors))]
        for detector in selected:
            for filtering in (False, True):
                for fmt in args.deployment_formats:
                    records.append(_deployment_record(args, detector, filtering, fmt))
    if args.stage in {"all", "report"}:
        records.append(StageRecord("REPORT", "report", command=_report_command(args), artifact_dir=str(suite_paths(args)["reports"])))
    return records


def write_run_manifest(args: argparse.Namespace, records: list[StageRecord], status: str = "planned") -> Path:
    paths = suite_paths(args)
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 2,
        "run_id": "ablation_deployment_suite",
        "status": status,
        "created_at": _now(),
        "repository": str(repo_root()),
        "python": sys.version,
        "platform": platform.platform(),
        "experimental_controls": {
            "data_variant": args.data_variant,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "device": args.device,
            "seed": args.seed,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "workers": args.workers,
            "conf": args.conf,
            "iou": args.iou,
            "tracker": "botsort",
            "vid_stride": args.vid_stride,
            "detector_eval_data": str(_detector_eval_data(args)) if _detector_eval_data(args) else None,
            "detector_eval_split": args.detector_eval_split,
            "pipeline_video": str(_pipeline_video(args)) if _pipeline_video(args) else None,
        },
        "detectors": [asdict(detector) for detector in DETECTOR_GRID],
        "pipeline_experiments": [asdict(item) for item in pipeline_grid()],
        "deployment_formats": list(args.deployment_formats),
        "records": [asdict(record) for record in records],
    }
    path = paths["manifests"] / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def run_suite(args: argparse.Namespace) -> int:
    for path in suite_paths(args).values():
        path.mkdir(parents=True, exist_ok=True)
    records = planned_records(args)
    manifest = write_run_manifest(args, records, status="planned")

    if args.dry_run:
        diagnostics = dry_run_diagnostics(args, records)
        (suite_paths(args)["manifests"] / "dry_run_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"manifest": str(manifest), **diagnostics}, indent=2))
        return 0

    preflight_errors = _preflight_errors(args)
    if preflight_errors:
        error_path = suite_paths(args)["manifests"] / "preflight_errors.json"
        error_path.write_text(json.dumps({"errors": preflight_errors}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "preflight_failed", "errors": preflight_errors, "error_report": str(error_path)}, indent=2))
        return 2

    status_by_key = _load_existing_statuses(suite_paths(args)["manifests"])
    executed: list[StageRecord] = []

    first_pass = [record for record in records if record.stage in {"detection", "detector_eval", "filtering"}]
    if args.stage == "deployment":
        first_pass = []
    if args.stage == "report":
        first_pass = []

    for record in first_pass:
        key = (record.stage, record.experiment_id)
        old_status = status_by_key.get(key)
        if args.resume and old_status == "success":
            record.status = "skipped"
            record.error_message = "Skipped by --resume because a successful manifest already exists."
            executed.append(record)
            continue
        if args.resume and old_status == "failed" and not args.rerun_failed:
            record.status = "skipped"
            record.error_message = "Skipped failed stage by --resume; pass --rerun-failed to retry."
            executed.append(record)
            continue

        completed = _execute_record(args, record)
        executed.append(completed)
        _write_stage_manifest(args, completed)

    aggregate_results(args)

    remaining: list[StageRecord] = []
    if args.stage in {"all", "deployment"}:
        remaining.extend(_deployment_records_from_shortlist(args))
    if args.stage in {"all", "report"}:
        remaining.append(StageRecord("REPORT", "report", command=_report_command(args), artifact_dir=str(suite_paths(args)["reports"])))

    for record in remaining:
        key = (record.stage, record.experiment_id)
        old_status = status_by_key.get(key)
        if args.resume and old_status == "success":
            record.status = "skipped"
            record.error_message = "Skipped by --resume because a successful manifest already exists."
            executed.append(record)
            continue
        if args.resume and old_status == "failed" and not args.rerun_failed:
            record.status = "skipped"
            record.error_message = "Skipped failed stage by --resume; pass --rerun-failed to retry."
            executed.append(record)
            continue
        completed = _execute_record(args, record)
        executed.append(completed)
        _write_stage_manifest(args, completed)

    aggregate_results(args)
    write_run_manifest(args, executed, status="completed")
    return 1 if any(record.status == "failed" for record in executed) else 0


def _deployment_records_from_shortlist(args: argparse.Namespace) -> list[StageRecord]:
    if args.deployment_mode == "all":
        selected_ids = [detector.experiment_id for detector in DETECTOR_GRID]
    else:
        shortlist_path = suite_paths(args)["aggregated"] / "deployment_shortlist.json"
        if shortlist_path.exists():
            payload = json.loads(shortlist_path.read_text(encoding="utf-8"))
            selected_ids = [str(item["experiment_id"]) for item in payload[: args.shortlist_size]]
        else:
            selected_ids = [detector.experiment_id for detector in DETECTOR_GRID[: args.shortlist_size]]
    detectors = [detector_by_id(experiment_id) for experiment_id in selected_ids]
    return [
        _deployment_record(args, detector, filtering_enabled, fmt)
        for detector in detectors
        for filtering_enabled in (False, True)
        for fmt in args.deployment_formats
    ]


def dry_run_diagnostics(args: argparse.Namespace, records: list[StageRecord]) -> dict[str, Any]:
    missing: list[str] = []
    warnings: list[str] = []
    if args.video and not _resolve(args.video).exists():
        missing.append(f"video: {args.video}")
    if args.pipeline_video and not _resolve(args.pipeline_video).exists():
        missing.append(f"pipeline_video: {args.pipeline_video}")
    eval_data = _detector_eval_data(args)
    if eval_data is not None and not eval_data.exists():
        missing.append(f"detector_eval_data: {eval_data}")
    if args.gt_labels and not _resolve(args.gt_labels).exists():
        missing.append(f"gt_labels: {args.gt_labels}")
    calibration_dataset = _calibration_dataset_yaml(args)
    if calibration_dataset and not calibration_dataset.exists():
        missing.append(f"calibration_dataset_yaml: {calibration_dataset}")
    if any(det.use_negative_attention for det in DETECTOR_GRID) and not args.negative_mask_root and not args.allow_dataset_negative_masks:
        warnings.append("Negative-attention training requires --negative-mask-root or --allow-dataset-negative-masks.")
    if args.stage in {"all", "filtering", "deployment"} and not _pipeline_video(args):
        warnings.append("Filtering/deployment evaluation requires --pipeline-video or legacy --video.")
    if args.max_parallel != 1:
        warnings.append("--max-parallel is recorded but execution remains sequential to respect GPU safety.")

    return {
        "expected_detector_runs": len(DETECTOR_GRID),
        "expected_detector_eval_runs": sum(1 for record in records if record.stage == "detector_eval"),
        "expected_pipeline_runs": len(pipeline_grid()),
        "expected_deployment_runs": sum(1 for record in records if record.stage == "deployment"),
        "expected_total_records": len(records),
        "missing": missing,
        "warnings": warnings,
        "planned_records": [asdict(record) for record in records],
    }


def aggregate_results(args: argparse.Namespace) -> dict[str, Any]:
    paths = suite_paths(args)
    paths["aggregated"].mkdir(parents=True, exist_ok=True)
    paths["plots"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    pipeline_rows = [_pipeline_summary_row(args, item) for item in pipeline_grid()]
    detector_rows = [_detector_summary_row(args, detector, pipeline_rows) for detector in DETECTOR_GRID]
    deployment_rows = _deployment_summary_rows(args)
    factorial = factorial_effects(detector_rows)
    shortlist = select_deployment_shortlist(detector_rows, args.shortlist_size)

    _write_csv(paths["aggregated"] / "detector_results.csv", detector_rows)
    _write_csv(paths["aggregated"] / "pipeline_results.csv", pipeline_rows)
    _write_csv(paths["aggregated"] / "deployment_results.csv", deployment_rows)
    (paths["aggregated"] / "factorial_effects.json").write_text(json.dumps(factorial, indent=2) + "\n", encoding="utf-8")
    (paths["aggregated"] / "deployment_shortlist.json").write_text(json.dumps(shortlist, indent=2) + "\n", encoding="utf-8")
    summary = {
        "detectors": detector_rows,
        "pipelines": pipeline_rows,
        "deployment": deployment_rows,
        "factorial_effects": factorial,
        "deployment_shortlist": shortlist,
    }
    (paths["aggregated"] / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_suite_plots(paths["plots"], detector_rows, pipeline_rows, deployment_rows, factorial)
    write_final_report(paths["reports"] / "final_report.md", args, detector_rows, pipeline_rows, deployment_rows, factorial, shortlist)
    return summary


def factorial_effects(rows: list[dict[str, Any]], metric: str = "map50") -> dict[str, Any]:
    by_id = {row["experiment_id"]: row for row in rows}
    values = {det.experiment_id: _as_float(by_id.get(det.experiment_id, {}).get(metric)) for det in DETECTOR_GRID}
    effects: dict[str, Any] = {"metric": metric, "available": all(value is not None for value in values.values())}
    if not effects["available"]:
        effects["reason"] = f"Metric {metric!r} is unavailable for at least one detector."
        effects["values"] = values
        return effects

    factors = {
        "cse": {det.experiment_id: det.use_cse for det in DETECTOR_GRID},
        "negative_attention": {det.experiment_id: det.use_negative_attention for det in DETECTOR_GRID},
        "geometry_attention": {det.experiment_id: det.use_geometry_attention for det in DETECTOR_GRID},
    }
    main: dict[str, float] = {}
    for name, enabled_by_id in factors.items():
        on = [values[exp_id] for exp_id, enabled in enabled_by_id.items() if enabled]
        off = [values[exp_id] for exp_id, enabled in enabled_by_id.items() if not enabled]
        main[name] = round(_mean(on) - _mean(off), 6)
    effects["main_effects"] = main
    effects["interpretation"] = _factorial_interpretation(main)
    return effects


def select_deployment_shortlist(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        f1 = _as_float(row.get("f1")) or 0.0
        map50 = _as_float(row.get("map50")) or 0.0
        map5095 = _as_float(row.get("map50_95")) or 0.0
        fps = _as_float(row.get("fps")) or 0.0
        size = _as_float(row.get("model_size_mb")) or 0.0
        score = f1 + map50 + map5095 + (min(fps, 100.0) / 1000.0) - (size / 10000.0)
        scored.append(
            {
                "experiment_id": row["experiment_id"],
                "case": row["case"],
                "selection_score": round(score, 6),
                "selection_rule": "f1 + mAP50 + mAP50-95 + capped_fps/1000 - model_size_mb/10000",
            }
        )
    return sorted(scored, key=lambda item: (item["selection_score"], item["experiment_id"]), reverse=True)[:limit]


def write_suite_plots(
    out: Path,
    detector_rows: list[dict[str, Any]],
    pipeline_rows: list[dict[str, Any]],
    deployment_rows: list[dict[str, Any]],
    factorial: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out / "plots_unavailable.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    _bar_plot(plt, out / "detector_map50.png", detector_rows, "experiment_id", "map50", "Detection Ablation: mAP50")
    _bar_plot(plt, out / "detector_map50_95.png", detector_rows, "experiment_id", "map50_95", "Detection Ablation: mAP50-95")
    _grouped_metric_plot(plt, out / "detector_precision_recall_f1.png", detector_rows, ["precision", "recall", "f1"], "Detector Metrics")
    _bar_plot(plt, out / "false_positives_per_frame.png", detector_rows, "experiment_id", "false_positives_per_frame", "False Positives per Frame")
    if factorial.get("available"):
        rows = [{"factor": key, "effect": value} for key, value in factorial.get("main_effects", {}).items()]
        _bar_plot(plt, out / "factorial_main_effects.png", rows, "factor", "effect", "Observed Main Effects")
    _filtering_plot(plt, out / "filtering_f1_comparison.png", pipeline_rows)
    if deployment_rows:
        _scatter_plot(plt, out / "deployment_accuracy_vs_fps.png", deployment_rows, "fps", "map50", "Deployment Accuracy vs FPS")
        _scatter_plot(plt, out / "deployment_size_vs_accuracy.png", deployment_rows, "model_size_mb", "map50", "Deployment Size vs Accuracy")


def write_final_report(
    path: Path,
    args: argparse.Namespace,
    detector_rows: list[dict[str, Any]],
    pipeline_rows: list[dict[str, Any]],
    deployment_rows: list[dict[str, Any]],
    factorial: dict[str, Any],
    shortlist: list[dict[str, Any]],
) -> None:
    lines = [
        "# YOLO26 Ablation and Deployment Benchmark Report",
        "",
        f"Generated: {_now()}",
        "",
        "## Experimental Setup",
        "",
        f"- Data variant: `{args.data_variant}`",
        f"- Tracker: `botsort`",
        f"- Seed: `{args.seed}`",
        f"- Image size: `{args.imgsz}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch: `{args.batch}`",
        "",
        "## Detector Ablation Results",
        "",
        _markdown_table(detector_rows, ["experiment_id", "case", "map50", "map50_95", "precision", "recall", "f1", "false_positives_per_frame", "status"]),
        "",
        "## Observed Factorial Effects",
        "",
        "These are observed single-run effects, not statistical significance claims.",
        "",
        "```json",
        json.dumps(factorial, indent=2),
        "```",
        "",
        "## Filtering Results",
        "",
        _markdown_table(pipeline_rows, ["experiment_id", "detector_id", "filtering_enabled", "precision", "recall", "f1", "false_positives_per_frame", "status"]),
        "",
        "## Deployment Shortlist",
        "",
        _markdown_table(shortlist, ["experiment_id", "case", "selection_score", "selection_rule"]),
        "",
        "## Deployment Benchmarks",
        "",
        _markdown_table(deployment_rows, ["deployment_id", "detector_id", "filtering_enabled", "format", "map50", "map50_95", "precision", "recall", "f1", "fps", "latency_ms", "model_size_mb", "status"]),
        "",
        "## Reproducibility",
        "",
        f"- Run manifest: `{suite_paths(args)['manifests'] / 'run_manifest.json'}`",
        f"- Aggregated artifacts: `{suite_paths(args)['aggregated']}`",
        f"- Plots: `{suite_paths(args)['plots']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute_record(args: argparse.Namespace, record: StageRecord) -> StageRecord:
    record.start_time = _now()
    start = time.perf_counter()
    log_path = suite_paths(args)["logs"] / f"{record.stage}_{record.experiment_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if record.stage == "report":
            aggregate_results(args)
            record.status = "success"
            record.returncode = 0
            record.completion_time = _now()
            record.duration_s = round(time.perf_counter() - start, 3)
            return record
        if record.stage == "deployment":
            completed = _execute_deployment_record(args, record, log_path)
            record.status = completed.status
            record.returncode = completed.returncode
            record.error_message = completed.error_message
            record.metadata.update(completed.metadata)
            record.completion_time = _now()
            record.duration_s = round(time.perf_counter() - start, 3)
            record.metrics = _metrics_for_record(args, record)
            return record
        env = _subprocess_env()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(record.command, cwd=repo_root(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            returncode = int(completed.returncode)
        record.returncode = returncode
        record.status = "success" if returncode == 0 else "failed"
        if returncode != 0:
            record.error_message = f"Command failed with return code {returncode}; see {log_path}"
    except Exception as exc:
        record.status = "failed"
        record.error_message = str(exc)
    record.completion_time = _now()
    record.duration_s = round(time.perf_counter() - start, 3)
    record.metrics = _metrics_for_record(args, record)
    return record


def _execute_deployment_record(args: argparse.Namespace, record: StageRecord, log_path: Path) -> StageRecord:
    env = _subprocess_env()
    export_dir = Path(record.metadata["export_dir"])
    fmt = str(record.metadata["format"])
    with log_path.open("w", encoding="utf-8") as log:
        log.write("== export ==\n")
        export_result = subprocess.run(record.command, cwd=repo_root(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if export_result.returncode != 0:
            record.status = "failed"
            record.returncode = int(export_result.returncode)
            record.error_message = f"Export failed with return code {export_result.returncode}; see {log_path}"
            return record
        exported_model = _exported_model_from_report(export_dir, fmt)
        if exported_model is None or not exported_model.exists():
            record.status = "failed"
            record.returncode = 1
            record.error_message = f"Export report did not contain a usable model for {fmt}; see {export_dir / 'export_report.json'}"
            return record
        benchmark_command = _deployment_audit_command(args, record, exported_model)
        record.metadata["exported_model"] = str(exported_model)
        record.metadata["benchmark_command"] = benchmark_command
        log.write("\n== audit benchmark ==\n")
        benchmark_result = subprocess.run(benchmark_command, cwd=repo_root(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        record.returncode = int(benchmark_result.returncode)
        record.status = "success" if benchmark_result.returncode == 0 else "failed"
        if benchmark_result.returncode != 0:
            record.error_message = f"Deployment audit failed with return code {benchmark_result.returncode}; see {log_path}"
        return record


def _write_stage_manifest(args: argparse.Namespace, record: StageRecord) -> Path:
    path = suite_paths(args)["manifests"] / f"{record.stage}_{record.experiment_id}.json"
    path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
    return path


def _load_existing_statuses(manifest_dir: Path) -> dict[tuple[str, str], str]:
    statuses: dict[tuple[str, str], str] = {}
    for path in manifest_dir.glob("*.json"):
        if path.name == "run_manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stage = payload.get("stage")
        exp_id = payload.get("experiment_id")
        if stage and exp_id:
            statuses[(stage, exp_id)] = str(payload.get("status", ""))
    return statuses


def _detector_record(args: argparse.Namespace, detector: DetectorExperiment) -> StageRecord:
    cmd = [
        sys.executable,
        "-m",
        "rbccps_od.training.ablation",
        args.data_variant,
        "--cases",
        detector.case,
        "--imgsz",
        str(args.imgsz),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--optimizer",
        args.optimizer,
        "--lr0",
        str(args.lr0),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--workers",
        str(args.workers),
        "--close-mosaic",
        str(args.close_mosaic),
        "--project",
        args.project,
        "--artifact-root",
        args.artifact_root,
        "--skip-existing",
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.cache:
        cmd.append("--cache")
    if args.negative_mask_root:
        cmd.extend(["--negative-mask-root", args.negative_mask_root])
    if args.negative_mask_loss_weight != 1.0:
        cmd.extend(["--negative-mask-loss-weight", str(args.negative_mask_loss_weight)])
    if args.allow_dataset_negative_masks:
        cmd.append("--allow-dataset-negative-masks")
    if args.original_image_dir:
        cmd.extend(["--original-image-dir", args.original_image_dir])
    if args.original_label_dir:
        cmd.extend(["--original-label-dir", args.original_label_dir])
    if args.original_dataset_root:
        cmd.extend(["--original-dataset-root", args.original_dataset_root])
    cmd.extend(["--link-mode", args.link_mode])
    if args.overwrite_original_view:
        cmd.append("--overwrite-original-view")
    return StageRecord(detector.experiment_id, "detection", command=cmd, artifact_dir=str(_detector_artifact_dir(args, detector)))


def _pipeline_record(args: argparse.Namespace, item: PipelineExperiment) -> StageRecord:
    detector = detector_by_id(item.detector_id)
    out = suite_paths(args)["pipeline"] / item.experiment_id
    video = _pipeline_video(args)
    cmd = [
        sys.executable,
        "-m",
        "evaluation.eval_pres.run_audit",
        "--video",
        str(video) if video else "",
        "--model",
        str(_detector_artifact_dir(args, detector) / "best.pt"),
        "--output-dir",
        str(out),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--imgsz",
        str(args.imgsz),
        "--device",
        str(args.device),
        "--tracker",
        "botsort",
        "--vid-stride",
        str(args.vid_stride),
    ]
    if not item.filtering_enabled:
        cmd.append("--disable-multicue")
    if args.no_save_video:
        cmd.append("--no-save-video")
    if args.gt_labels:
        cmd.extend(["--gt-labels", args.gt_labels])
    if args.gt_status_file:
        cmd.extend(["--gt-status-file", args.gt_status_file])
    if detector.use_cse:
        cmd.append("--use-cse")
    if detector.use_negative_attention:
        cmd.append("--use-negative-attention")
    if detector.use_geometry_attention:
        cmd.append("--use-geometry-attention")
    return StageRecord(item.experiment_id, "filtering", command=cmd, artifact_dir=str(out))


def _detector_eval_record(args: argparse.Namespace, detector: DetectorExperiment) -> StageRecord:
    out = _detector_eval_dir(args, detector)
    data = _detector_eval_data(args)
    cmd = [
        sys.executable,
        "-m",
        "rbccps_od.training.validate",
        "--model",
        str(_detector_artifact_dir(args, detector) / "best.pt"),
        "--data",
        str(data) if data else "",
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--split",
        args.detector_eval_split,
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--output-json",
        str(out / "metrics.json"),
    ]
    return StageRecord(detector.experiment_id, "detector_eval", command=cmd, artifact_dir=str(out))


def _deployment_record(args: argparse.Namespace, detector: DetectorExperiment, filtering_enabled: bool, fmt: str) -> StageRecord:
    suffix = "FILTER" if filtering_enabled else "NOFILTER"
    deployment_id = f"{detector.experiment_id}_{suffix}_{fmt.upper()}"
    export_dir = suite_paths(args)["exports"] / deployment_id
    bench_dir = suite_paths(args)["benchmarks"] / deployment_id
    video = _pipeline_video(args)
    cmd = [
        sys.executable,
        "scripts/export_quantized_models.py",
        "--weights",
        str(_detector_artifact_dir(args, detector) / "best.pt"),
        "--out",
        str(export_dir),
        "--imgsz",
        str(args.imgsz),
        "--formats",
        fmt,
    ]
    calibration_dataset = _calibration_dataset_yaml(args)
    if calibration_dataset:
        cmd.extend(["--data", str(calibration_dataset)])
    cmd.extend(
        [
            "&&",
            sys.executable,
            "scripts/benchmark_quantized_models.py",
            "--export-dir",
            str(export_dir),
            "--baseline",
            str(_detector_artifact_dir(args, detector) / "best.pt"),
            "--video",
            str(video) if video else "",
            "--out",
            str(bench_dir),
            "--preset",
            args.deployment_preset,
            "--conf",
            str(args.conf),
        ]
    )
    if args.deployment_max_frames is not None:
        cmd.extend(["--max-frames", str(args.deployment_max_frames)])
    return StageRecord(deployment_id, "deployment", command=cmd, artifact_dir=str(bench_dir))


def _report_command(args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-m", "rbccps_od.training.benchmark_suite", "--stage", "report", "--output-root", args.output_root, "--dry-run"]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo_root() / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else f"{src}{os.pathsep}{current}"
    return env


def _metrics_for_record(args: argparse.Namespace, record: StageRecord) -> dict[str, Any]:
    if record.stage == "filtering":
        return _extract_audit_metrics(Path(record.artifact_dir))
    if record.stage == "detection":
        return _extract_training_metrics(Path(record.artifact_dir))
    if record.stage == "detector_eval":
        return _extract_detector_eval_metrics(Path(record.artifact_dir))
    if record.stage == "deployment":
        return _extract_deployment_metrics(Path(record.artifact_dir))
    return {}


def _detector_summary_row(args: argparse.Namespace, detector: DetectorExperiment, pipeline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    no_filter_id = next(item.experiment_id for item in pipeline_grid() if item.detector_id == detector.experiment_id and not item.filtering_enabled)
    pipeline = next((row for row in pipeline_rows if row["experiment_id"] == no_filter_id), {})
    training = _extract_training_metrics(_detector_artifact_dir(args, detector))
    evaluation = _extract_detector_eval_metrics(_detector_eval_dir(args, detector))
    return {
        "experiment_id": detector.experiment_id,
        "case": detector.case,
        "use_cse": detector.use_cse,
        "use_negative_attention": detector.use_negative_attention,
        "use_geometry_attention": detector.use_geometry_attention,
        "checkpoint": str(_detector_artifact_dir(args, detector) / "best.pt"),
        "status": "success" if evaluation else ("trained" if (_detector_artifact_dir(args, detector) / "best.pt").exists() else "missing_checkpoint"),
        "map50": evaluation.get("map50") or training.get("map50"),
        "map50_95": evaluation.get("map50_95") or training.get("map50_95"),
        "precision": evaluation.get("precision") or training.get("precision"),
        "recall": evaluation.get("recall") or training.get("recall"),
        "f1": evaluation.get("f1") or training.get("f1"),
        "false_positives_per_frame": None,
        "pipeline_map50": pipeline.get("map50"),
        "pipeline_map50_95": pipeline.get("map50_95"),
        "pipeline_precision": pipeline.get("precision"),
        "pipeline_recall": pipeline.get("recall"),
        "pipeline_f1": pipeline.get("f1"),
        "pipeline_false_positives_per_frame": pipeline.get("false_positives_per_frame"),
        "fps": pipeline.get("fps"),
        "model_size_mb": _file_size_mb(_detector_artifact_dir(args, detector) / "best.pt"),
    }


def _pipeline_summary_row(args: argparse.Namespace, item: PipelineExperiment) -> dict[str, Any]:
    metrics = _extract_audit_metrics(suite_paths(args)["pipeline"] / item.experiment_id)
    return {
        "experiment_id": item.experiment_id,
        "detector_id": item.detector_id,
        "filtering_enabled": item.filtering_enabled,
        "status": "success" if metrics else "missing",
        **metrics,
    }


def _deployment_summary_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in suite_paths(args)["benchmarks"].glob("*"):
        if not path.is_dir():
            continue
        metrics = _extract_deployment_metrics(path)
        parts = path.name.split("_")
        rows.append(
            {
                "deployment_id": path.name,
                "detector_id": parts[0] if parts else "",
                "filtering_enabled": "_FILTER_" in f"_{path.name}_",
                "format": "_".join(parts[2:]).lower() if len(parts) > 2 else "",
                "status": "success" if metrics else "missing",
                **metrics,
            }
        )
    return sorted(rows, key=lambda row: row["deployment_id"])


def _extract_audit_metrics(out: Path) -> dict[str, Any]:
    report = out / "audit_report.json"
    if not report.exists():
        return {}
    payload = json.loads(report.read_text(encoding="utf-8"))
    eval_metrics = payload.get("evaluation_metrics", {}) or {}
    detection = eval_metrics.get("detection", {}) if isinstance(eval_metrics, dict) else {}
    frame = eval_metrics.get("frame_detection", {}) if isinstance(eval_metrics, dict) else {}
    before_after = eval_metrics.get("before_after_filtering", {}) if isinstance(eval_metrics, dict) else {}
    context = eval_metrics.get("evaluation_context", {}) if isinstance(eval_metrics, dict) else {}
    video = payload.get("video", {}) or {}
    return {
        "map50": detection.get("AP@0.5"),
        "map50_95": detection.get("mAP@0.5:0.95"),
        "precision": frame.get("precision"),
        "recall": frame.get("recall"),
        "f1": frame.get("f1"),
        "false_positives_per_frame": _fp_per_frame(frame, context),
        "raw_prediction_instances": context.get("raw_target_prediction_instances"),
        "filtered_prediction_instances": context.get("filtered_target_prediction_instances"),
        "frames_evaluated": context.get("frames_evaluated") or video.get("frame_count"),
        "before_after_filtering": before_after,
    }


def _extract_training_metrics(artifact_dir: Path) -> dict[str, Any]:
    results = artifact_dir / "results.csv"
    if not results.exists():
        return {}
    rows = list(csv.DictReader(results.open(encoding="utf-8")))
    if not rows:
        return {}
    row = rows[-1]
    return {
        "precision": _first_numeric(row, ("metrics/precision(B)", "precision")),
        "recall": _first_numeric(row, ("metrics/recall(B)", "recall")),
        "map50": _first_numeric(row, ("metrics/mAP50(B)", "mAP50")),
        "map50_95": _first_numeric(row, ("metrics/mAP50-95(B)", "mAP50-95")),
    }


def _extract_detector_eval_metrics(artifact_dir: Path) -> dict[str, Any]:
    metrics_path = artifact_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {}) or {}
    return {
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "map50": metrics.get("map50"),
        "map50_95": metrics.get("map50_95"),
    }


def _extract_deployment_metrics(out: Path) -> dict[str, Any]:
    report = out / "benchmark_report.json"
    if not report.exists():
        return {}
    payload = json.loads(report.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not results:
        return {}
    deployed = next((row for row in results if row.get("label") != "baseline_pt" and row.get("status") == "success"), results[0])
    metrics = deployed.get("evaluation_metrics", {}) or {}
    return {
        "fps": deployed.get("effective_fps"),
        "latency_ms": deployed.get("per_frame_avg_total_ms"),
        "model_size_mb": deployed.get("model_size_mb"),
        "precision": _metric_value(metrics, "Detection/Precision @0.5"),
        "recall": _metric_value(metrics, "Detection/Recall @0.5"),
        "f1": _metric_value(metrics, "Detection/F1 @0.5"),
        "map50": _metric_value(metrics, "Detection/AP@0.5"),
        "map50_95": _metric_value(metrics, "Detection/mAP@0.5:0.95"),
    }


def _detector_artifact_dir(args: argparse.Namespace, detector: DetectorExperiment) -> Path:
    experiment = EXPERIMENT_BY_NAME[args.data_variant]
    case = CASE_BY_NAME[detector.case]
    return _resolve(args.artifact_root) / _artifact_name_for(experiment, case)


def _detector_eval_dir(args: argparse.Namespace, detector: DetectorExperiment) -> Path:
    return suite_paths(args)["detector_eval"] / detector.experiment_id


def _detector_eval_data(args: argparse.Namespace) -> Path | None:
    if args.detector_eval_data:
        return _resolve(args.detector_eval_data)
    experiment = EXPERIMENT_BY_NAME[args.data_variant]
    if experiment.dataset_yaml:
        return _resolve(experiment.dataset_yaml)
    if experiment.uses_original_view:
        return _resolve(args.original_dataset_root) / "dataset.yaml"
    return None


def _pipeline_video(args: argparse.Namespace) -> Path | None:
    if args.pipeline_video:
        return _resolve(args.pipeline_video)
    if args.video:
        return _resolve(args.video)
    return None


def _calibration_dataset_yaml(args: argparse.Namespace) -> Path | None:
    value = getattr(args, "calibration_dataset_yaml", None)
    return _resolve(value) if value else None


def detector_by_id(experiment_id: str) -> DetectorExperiment:
    for detector in DETECTOR_GRID:
        if detector.experiment_id == experiment_id:
            return detector
    raise KeyError(experiment_id)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fp_per_frame(frame: dict[str, Any], context: dict[str, Any]) -> float | None:
    fp = _as_float(frame.get("fp"))
    frames = _as_float(context.get("frames_evaluated"))
    if fp is None or not frames:
        return None
    return round(fp / frames, 6)


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("value")
    return value


def _first_numeric(row: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in row:
            value = _as_float(row[key])
            if value is not None:
                return value
    return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float | None]) -> float:
    actual = [float(value) for value in values if value is not None]
    return sum(actual) / len(actual) if actual else 0.0


def _factorial_interpretation(main: dict[str, float]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in main.items():
        if value > 0:
            result[key] = "observed positive effect"
        elif value < 0:
            result[key] = "observed negative effect"
        else:
            result[key] = "no observed effect"
    return result


def _file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 4)
    return round(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024 * 1024), 4)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _bar_plot(plt: Any, path: Path, rows: list[dict[str, Any]], label_key: str, value_key: str, title: str) -> None:
    labels = [str(row.get(label_key, "")) for row in rows]
    values = [_as_float(row.get(value_key)) or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#2f6f73")
    ax.set_title(title)
    ax.set_ylabel(value_key)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _grouped_metric_plot(plt: Any, path: Path, rows: list[dict[str, Any]], metrics: list[str], title: str) -> None:
    labels = [row["experiment_id"] for row in rows]
    x_positions = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, metric in enumerate(metrics):
        values = [_as_float(row.get(metric)) or 0.0 for row in rows]
        offsets = [x + (idx - 1) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=metric)
    ax.set_xticks(x_positions, labels=labels)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _filtering_plot(plt: Any, path: Path, rows: list[dict[str, Any]]) -> None:
    by_detector: dict[str, dict[bool, dict[str, Any]]] = {}
    for row in rows:
        by_detector.setdefault(row["detector_id"], {})[bool(row["filtering_enabled"])] = row
    plot_rows = [
        {
            "detector_id": detector_id,
            "no_filter_f1": _as_float(states.get(False, {}).get("f1")) or 0.0,
            "filter_f1": _as_float(states.get(True, {}).get("f1")) or 0.0,
        }
        for detector_id, states in sorted(by_detector.items())
    ]
    labels = [row["detector_id"] for row in plot_rows]
    x_positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([x - 0.18 for x in x_positions], [row["no_filter_f1"] for row in plot_rows], width=0.36, label="no filtering")
    ax.bar([x + 0.18 for x in x_positions], [row["filter_f1"] for row in plot_rows], width=0.36, label="filtering")
    ax.set_xticks(x_positions, labels=labels)
    ax.set_title("Filtering Effect on F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scatter_plot(plt: Any, path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    x = [_as_float(row.get(x_key)) or 0.0 for row in rows]
    y = [_as_float(row.get(y_key)) or 0.0 for row in rows]
    labels = [row.get("deployment_id", "") for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="#6f4e7c")
    for x_value, y_value, label in zip(x, y, labels):
        ax.annotate(label, (x_value, y_value), fontsize=6, alpha=0.8)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_suite(parse_args(argv)))


if __name__ == "__main__":
    main()
