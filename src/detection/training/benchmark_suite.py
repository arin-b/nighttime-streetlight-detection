from __future__ import annotations

import argparse
import csv
import importlib.util
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
    parser.add_argument("--detector-eval-conf", type=float, default=0.001,
                        help="Low confidence threshold for an unbiased YOLO mAP curve.")
    parser.add_argument("--detector-eval-iou", type=float, default=0.7,
                        help="NMS IoU used for detector-only validation.")
    parser.add_argument("--gt-labels", help="YOLO label directory for held-out evaluation.")
    parser.add_argument("--gt-status-file", help="Optional status-classification ground truth CSV.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--no-save-video", dest="no_save_video", action="store_true", default=True,
                        help="Skip annotated video files during benchmarking to save time and disk.")
    parser.add_argument("--save-video", dest="no_save_video", action="store_false",
                        help="Also write annotated pipeline videos.")

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
    parser.add_argument("--shortlist-size", type=int, default=8, help="Maximum detector/filtering pipelines to export.")
    parser.add_argument("--min-precision", type=float, default=0.0, help="Minimum held-out precision for deployment selection.")
    parser.add_argument("--min-recall", type=float, default=0.0, help="Minimum held-out recall for deployment selection.")
    parser.add_argument("--deployment-device", default="cpu", help="Device for host-side exported-model evaluation.")
    args = parser.parse_args(argv)

    if args.config:
        args = _merge_config_args(args, Path(args.config))
    return args


def _merge_config_args(args: argparse.Namespace, config_path: Path) -> argparse.Namespace:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("--config JSON must contain an object of argument names to values.")
    unknown = [key for key in config if not hasattr(args, key.replace("-", "_"))]
    if unknown:
        raise ValueError(f"Unknown --config option(s): {', '.join(sorted(unknown))}")
    for key, value in config.items():
        setattr(args, key.replace("-", "_"), value)
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
        selected = pipelines if args.deployment_mode == "all" else pipelines[: args.shortlist_size]
        for pipeline in selected:
            for fmt in args.deployment_formats:
                records.append(_deployment_record(args, pipeline, fmt))
    if args.stage in {"all", "report"}:
        records.append(StageRecord("REPORT", "report", command=_report_command(args), artifact_dir=str(suite_paths(args)["reports"])))
    return records


def write_run_manifest(args: argparse.Namespace, records: list[StageRecord], status: str = "planned") -> Path:
    paths = suite_paths(args)
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 2,
        "run_id": "ablation_deployment_suite",
        "run_signature": _run_signature(args),
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
            "detector_eval_conf": args.detector_eval_conf,
            "detector_eval_iou": args.detector_eval_iou,
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
    if args.stage == "report":
        _restore_report_controls(args)
    paths = suite_paths(args)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if args.data_variant not in EXPERIMENT_BY_NAME:
        print(f"Unknown data variant: {args.data_variant}. Choose from {sorted(EXPERIMENT_BY_NAME)}")
        return 2
    records = planned_records(args)
    old_statuses = _load_existing_statuses(paths["manifests"])
    if old_statuses and not args.resume and args.stage != "report":
        print("Existing stage results found under --output-root. Use --resume or choose a new output root.")
        return 2
    if args.resume and old_statuses:
        mismatch = _stage_signature_mismatches(args, paths["manifests"])
        if mismatch:
            print(f"Resume refused: {', '.join(mismatch[:5])} changed for an existing stage. Use a new --output-root.")
            return 2
    manifest = write_run_manifest(args, records, status="planned")

    if args.dry_run:
        diagnostics = dry_run_diagnostics(args, records)
        (paths["manifests"] / "dry_run_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"manifest": str(manifest), **diagnostics}, indent=2))
        return 0

    preflight_errors = _preflight_errors(args)
    if preflight_errors:
        error_path = paths["manifests"] / "preflight_errors.json"
        error_path.write_text(json.dumps({"errors": preflight_errors}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "preflight_failed", "errors": preflight_errors, "error_report": str(error_path)}, indent=2))
        return 2

    executed: list[StageRecord] = []
    first_pass = [record for record in records if record.stage in {"detection", "detector_eval", "filtering"}]
    for record in first_pass:
        if _should_skip_record(args, record, old_statuses):
            record.status = "skipped"
            record.error_message = "Successful stage and artifact verified by --resume."
            executed.append(record)
            continue
        if args.resume and old_statuses.get((record.stage, record.experiment_id)) == "failed" and not args.rerun_failed:
            print(f"Resume stopped at failed {record.stage}/{record.experiment_id}; use --rerun-failed after correcting the error.")
            write_run_manifest(args, executed, status="failed")
            return 2
        if args.resume and record.stage == "detection":
            record.command.append("--resume-training")
        completed = _execute_record(args, record)
        executed.append(completed)
        _write_stage_manifest(args, completed)
        if completed.status != "success":
            aggregate_results(args)
            write_run_manifest(args, executed, status="failed")
            return 1

    aggregate_results(args)
    remaining: list[StageRecord] = []
    if args.stage in {"all", "deployment"}:
        remaining.extend(_deployment_records_from_shortlist(args))
        if not remaining:
            print("Deployment stopped: no eligible pipeline had complete held-out metrics and timing.")
            write_run_manifest(args, executed, status="failed")
            return 2
    if args.stage in {"all", "report"}:
        remaining.append(StageRecord("REPORT", "report", command=_report_command(args), artifact_dir=str(paths["reports"])))

    deployment_failed = False
    for record in remaining:
        if _should_skip_record(args, record, old_statuses):
            record.status = "skipped"
            record.error_message = "Successful stage and artifact verified by --resume."
            executed.append(record)
            continue
        if args.resume and old_statuses.get((record.stage, record.experiment_id)) == "failed" and not args.rerun_failed:
            print(f"Resume stopped at failed {record.stage}/{record.experiment_id}; use --rerun-failed after correcting the error.")
            write_run_manifest(args, executed, status="failed")
            return 2
        completed = _execute_record(args, record)
        executed.append(completed)
        _write_stage_manifest(args, completed)
        if completed.status != "success":
            deployment_failed = True
            if record.stage != "deployment":
                break

    aggregate_results(args)
    write_run_manifest(args, executed, status="completed_with_errors" if deployment_failed else "completed")
    return 1 if deployment_failed else 0


def _restore_report_controls(args: argparse.Namespace) -> None:
    """Use the recorded settings when rebuilding a report from a staged run."""
    manifest_dir = suite_paths(args)["manifests"]
    stage_keys = (
        ("detection", {
            "data_variant", "model", "imgsz", "epochs", "batch", "device", "seed",
            "optimizer", "lr0", "weight_decay", "patience", "workers", "cache",
            "close_mosaic", "project", "artifact_root", "negative_mask_root",
            "negative_mask_loss_weight", "original_image_dir", "original_label_dir",
            "original_dataset_root", "detector_eval_data", "detector_eval_split",
            "detector_eval_conf", "detector_eval_iou",
        }),
        ("filtering", {"conf", "iou", "vid_stride", "gt_labels", "gt_status_file", "no_save_video"}),
        ("deployment", {
            "calibration_dataset_yaml", "deployment_formats", "deployment_mode",
            "shortlist_size", "min_precision", "min_recall", "deployment_device",
        }),
    )
    for stage, keys in stage_keys:
        path = next(iter(sorted(manifest_dir.glob(f"{stage}_*.json"))), None)
        if path is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        signature = payload.get("run_signature", {})
        for key in keys:
            if key in signature and hasattr(args, key):
                setattr(args, key, signature[key])


def _deployment_records_from_shortlist(args: argparse.Namespace) -> list[StageRecord]:
    if args.deployment_mode == "all":
        selected = list(pipeline_grid())
    else:
        shortlist_path = suite_paths(args)["aggregated"] / "deployment_shortlist.json"
        if not shortlist_path.exists():
            return []
        payload = json.loads(shortlist_path.read_text(encoding="utf-8"))
        pipeline_by_id = {pipeline.experiment_id: pipeline for pipeline in pipeline_grid()}
        selected = [
            pipeline_by_id[item["experiment_id"]]
            for item in payload[: args.shortlist_size]
            if item.get("experiment_id") in pipeline_by_id
        ]
    return [_deployment_record(args, pipeline, fmt) for pipeline in selected for fmt in args.deployment_formats]


def _run_signature(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "data_variant", "model", "imgsz", "epochs", "batch", "device", "seed",
        "optimizer", "lr0", "weight_decay", "patience", "workers", "cache",
        "close_mosaic", "project", "artifact_root", "negative_mask_root",
        "negative_mask_loss_weight", "allow_dataset_negative_masks",
        "original_image_dir", "original_label_dir", "original_dataset_root",
        "link_mode", "detector_eval_data", "detector_eval_split",
        "detector_eval_conf", "detector_eval_iou", "gt_labels",
        "gt_status_file", "conf", "iou", "vid_stride", "no_save_video",
        "calibration_dataset_yaml", "deployment_formats", "deployment_mode",
        "shortlist_size", "min_precision", "min_recall", "deployment_device",
    )
    signature = {key: getattr(args, key) for key in keys}
    video = _pipeline_video(args)
    signature["effective_pipeline_video"] = str(video) if video else None
    return signature


def _stage_signature_mismatches(args: argparse.Namespace, manifest_dir: Path) -> list[str]:
    current = _run_signature(args)
    mismatches: list[str] = []
    shared = {"data_variant", "artifact_root", "imgsz"}
    train = {
        "model", "epochs", "batch", "device", "seed", "optimizer", "lr0",
        "weight_decay", "patience", "workers", "cache", "close_mosaic",
        "project", "negative_mask_root", "negative_mask_loss_weight",
        "allow_dataset_negative_masks", "original_image_dir",
        "original_label_dir", "original_dataset_root", "link_mode",
    }
    eval_keys = {"detector_eval_data", "detector_eval_split", "detector_eval_conf", "detector_eval_iou"}
    video_keys = {
        "effective_pipeline_video", "gt_labels", "gt_status_file",
        "conf", "iou", "vid_stride", "no_save_video",
    }
    deployment_keys = {"calibration_dataset_yaml", "deployment_device"}
    for path in sorted(manifest_dir.glob("*.json")):
        if path.name in {"run_manifest.json", "dry_run_diagnostics.json", "preflight_errors.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        stage = payload.get("stage")
        if not stage or stage == "report":
            continue
        previous = payload.get("run_signature")
        if not isinstance(previous, dict):
            mismatches.append(f"{path.name}: missing provenance")
            continue
        keys = set(shared)
        if stage == "detection" and args.stage in {"all", "detection"}:
            keys.update(train)
        if stage == "detector_eval" and args.stage in {"all", "detection"}:
            keys.update(eval_keys)
        if stage == "filtering" and args.stage in {"all", "filtering", "deployment"}:
            keys.update(video_keys)
        if stage == "deployment" and args.stage in {"all", "deployment"}:
            keys.update(video_keys | deployment_keys)
        mismatches.extend(
            f"{path.name}: {key}"
            for key in sorted(keys)
            if previous.get(key) != current.get(key)
        )
    return mismatches


def _should_skip_record(
    args: argparse.Namespace,
    record: StageRecord,
    old_statuses: dict[tuple[str, str], str],
) -> bool:
    if not args.resume or old_statuses.get((record.stage, record.experiment_id)) != "success":
        return False
    artifact = Path(record.artifact_dir)
    if record.stage == "detection":
        return (artifact / "best.pt").is_file()
    if record.stage == "detector_eval":
        return (artifact / "metrics.json").is_file()
    if record.stage == "filtering":
        return (artifact / "audit_report.json").is_file()
    if record.stage == "deployment":
        timing_path = artifact / "benchmark_timing.json"
        if not (artifact / "audit_report.json").is_file() or not timing_path.is_file():
            return False
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        exported = Path(timing.get("exported_model") or "")
        checkpoint = Path(record.metadata.get("checkpoint") or "")
        return (
            bool(timing.get("exported_model")) and exported.exists()
            and checkpoint.is_file() and exported.stat().st_mtime >= checkpoint.stat().st_mtime
        )
    if record.stage == "report":
        return (artifact / "final_report.md").is_file()
    return False

def _check_dataset_split(data: Path, split: str) -> list[str]:
    try:
        import yaml
        payload = yaml.safe_load(data.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return [f"Invalid dataset YAML object: {data}"]
        root = Path(str(payload.get("path") or data.parent)).expanduser()
        if not root.is_absolute():
            root = data.parent / root
        paths = payload.get(split)
        if not paths:
            return [f"Dataset YAML has no {split} split: {data}"]
        paths = paths if isinstance(paths, list) else [paths]
        errors: list[str] = []
        for name in paths:
            image_path = Path(str(name))
            if not image_path.is_absolute():
                image_path = root / image_path
            if not image_path.exists():
                errors.append(f"Dataset {split} path is missing: {image_path}")
            elif image_path.is_dir() and not any(
                file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                for file in image_path.iterdir()
            ):
                errors.append(f"Dataset {split} has no images: {image_path}")
        return errors
    except Exception as exc:
        return [f"Could not inspect dataset YAML {data}: {exc}"]


def _check_video_labels(video: Path, labels: Path, stride: int) -> list[str]:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            readable, _ = cap.read()
        finally:
            cap.release()
    except Exception as exc:
        return [f"Could not inspect video {video}: {exc}"]
    if frame_count <= 0 or not readable:
        return [f"Video has no readable first frame: {video}"]
    expected = range(1, frame_count + 1, stride)
    missing = [idx for idx in expected if not (labels / f"frame_{idx:06d}.txt").is_file()]
    if missing:
        return [
            f"Video labels missing for {len(missing)} sampled frames (first indices: {missing[:5]}). "
            "The audit treats missing frame labels as empty ground truth."
        ]
    return []


def _preflight_errors(args: argparse.Namespace) -> list[str]:
    """Validate inputs before the first expensive stage starts."""
    errors: list[str] = []
    if args.stage == "report":
        return errors
    if args.data_variant not in EXPERIMENT_BY_NAME:
        errors.append(f"Unknown data variant: {args.data_variant}")
        return errors
    if not 1 <= args.shortlist_size <= len(pipeline_grid()):
        errors.append("--shortlist-size must be between 1 and 16.")
    if not args.deployment_formats:
        errors.append("At least one --deployment-formats value is required.")
    if not 0.0 <= args.min_precision <= 1.0 or not 0.0 <= args.min_recall <= 1.0:
        errors.append("--min-precision and --min-recall must be between 0 and 1.")
    if not 0.0 <= args.detector_eval_conf <= 1.0 or not 0.0 <= args.detector_eval_iou <= 1.0:
        errors.append("Detector evaluation confidence and IoU must be between 0 and 1.")
    if args.imgsz <= 0 or args.batch <= 0 or args.epochs <= 0 or args.vid_stride <= 0:
        errors.append("Image size, batch, epochs, and video stride must be positive.")
    if args.max_parallel != 1:
        errors.append("--max-parallel must be 1; parallel GPU execution is not implemented.")
    if importlib.util.find_spec("ultralytics") is None:
        errors.append(f"Ultralytics is unavailable in {sys.executable}; activate the project environment.")
    if importlib.util.find_spec("torch") is None:
        errors.append(f"PyTorch is unavailable in {sys.executable}; activate the project environment.")
    elif args.device != "cpu" and args.stage in {"all", "detection", "filtering"}:
        try:
            import torch
            device = str(args.device).lower()
            if device == "mps":
                if not torch.backends.mps.is_available():
                    errors.append("Requested MPS device is unavailable.")
            else:
                selected = device.removeprefix("cuda:").split(",")
                if not selected or any(not item.isdigit() for item in selected):
                    errors.append(f"Unrecognized training device: {args.device}")
                elif not torch.cuda.is_available():
                    errors.append(f"CUDA device {args.device} is unavailable to {sys.executable}.")
                else:
                    count = torch.cuda.device_count()
                    unavailable = [item for item in selected if int(item) >= count]
                    if unavailable:
                        errors.append(f"Requested CUDA index {unavailable} is unavailable; device count is {count}.")
        except Exception as exc:
            errors.append(f"Could not check training device availability: {exc}")

    if args.stage in {"all", "detection"}:
        from rbccps_od.training.yolo26m_finetune import resolve_yolo26m_model
        model = resolve_yolo26m_model(args.model)
        if not Path(model).is_file():
            errors.append(f"Local starting checkpoint is missing: {model}")
        if not args.resume:
            existing = [
                str(_detector_artifact_dir(args, det) / "best.pt")
                for det in DETECTOR_GRID
                if (_detector_artifact_dir(args, det) / "best.pt").is_file()
            ]
            if existing:
                errors.append(f"Existing ablation checkpoints would be overwritten: {existing[:3]}. Use a new --artifact-root.")
        experiment = EXPERIMENT_BY_NAME[args.data_variant]
        training_images: list[tuple[Path, str]] = []
        if experiment.uses_original_view:
            from rbccps_od.training.original_dataset import collect_images, infer_split, matching_label
            image_dir = _resolve(args.original_image_dir)
            label_dir = _resolve(args.original_label_dir) if args.original_label_dir else image_dir.parent / "labels"
            if not image_dir.is_dir() or not label_dir.is_dir():
                errors.append(f"Original images or labels directory missing: {image_dir}, {label_dir}")
            else:
                images = collect_images(image_dir)
                if not images:
                    errors.append(f"No training images found in {image_dir}")
                splits: set[str] = set()
                for image in images:
                    try:
                        split = infer_split(image)
                        matching_label(image, label_dir)
                        splits.add(split)
                        training_images.append((image, split))
                    except (ValueError, FileNotFoundError) as exc:
                        errors.append(str(exc))
                        if len(errors) >= 20:
                            break
                if splits != {"train", "valid", "test"}:
                    errors.append(f"Original dataset needs train, valid and test splits; found {sorted(splits)}")
        else:
            dataset_yaml = _resolve(experiment.dataset_yaml)
            if not dataset_yaml.is_file():
                errors.append(f"Training dataset YAML is missing: {dataset_yaml}")

        if not args.negative_mask_root and not args.allow_dataset_negative_masks:
            errors.append("Four negative-attention runs require --negative-mask-root with training-image masks.")
        if experiment.uses_original_view and args.allow_dataset_negative_masks and not args.negative_mask_root:
            errors.append("The original YOLO dataset does not provide batch negative masks; pass --negative-mask-root.")
        if args.negative_mask_root:
            mask_root = _resolve(args.negative_mask_root)
            if not mask_root.is_dir():
                errors.append(f"Negative-mask directory is missing: {mask_root}")
            elif experiment.uses_original_view and training_images:
                suffixes = (".png", ".jpg", ".jpeg", ".npy", ".pt")
                missing = []
                for image, split in training_images:
                    if split != "train":
                        continue
                    candidates = [
                        path for suffix in suffixes
                        for path in (mask_root / f"{image.stem}{suffix}", mask_root / split / f"{image.stem}{suffix}")
                    ]
                    if not any(path.is_file() for path in candidates):
                        missing.append(image.name)
                if missing:
                    errors.append(
                        f"Negative masks missing for {len(missing)} training images (first: {missing[:3]}). "
                        "Missing masks otherwise become zero targets."
                    )

        data = _detector_eval_data(args)
        if args.detector_eval_data and (data is None or not data.is_file()):
            errors.append(f"Detector held-out YAML is missing: {data}")
        elif not experiment.uses_original_view and (data is None or not data.is_file()):
            errors.append(f"Detector held-out YAML is missing: {data}")
        elif data is not None and data.is_file():
            errors.extend(_check_dataset_split(data, args.detector_eval_split))
            if not experiment.uses_original_view:
                errors.extend(_check_dataset_split(data, "train"))
                errors.extend(_check_dataset_split(data, "val"))

    if args.stage in {"all", "filtering", "deployment"}:
        video = _pipeline_video(args)
        if video is None or not video.is_file():
            errors.append(f"Held-out pipeline video is missing: {video}")
        labels = _resolve(args.gt_labels) if args.gt_labels else None
        if labels is None or not labels.is_dir() or not any(labels.glob("*.txt")):
            errors.append(f"Frame-level video ground-truth labels are missing or empty: {labels}")
        elif video is not None and video.is_file():
            errors.extend(_check_video_labels(video, labels, args.vid_stride))
        if args.stage in {"filtering", "deployment"}:
            if args.stage == "filtering":
                required = [det.experiment_id for det in DETECTOR_GRID]
            elif args.deployment_mode == "all":
                required = [det.experiment_id for det in DETECTOR_GRID]
            else:
                shortlist_path = suite_paths(args)["aggregated"] / "deployment_shortlist.json"
                if not shortlist_path.is_file():
                    errors.append("No prior filtering shortlist found; run --stage filtering first.")
                    required = []
                else:
                    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
                    required = sorted({str(item["detector_id"]) for item in shortlist})
            for exp_id in required:
                checkpoint = _detector_artifact_dir(args, detector_by_id(exp_id)) / "best.pt"
                if not checkpoint.is_file():
                    errors.append(f"Trained detector checkpoint missing: {checkpoint}")
        if args.stage == "deployment":
            if args.deployment_mode == "all":
                selected_pipelines = list(pipeline_grid())
            else:
                shortlist_path = suite_paths(args)["aggregated"] / "deployment_shortlist.json"
                if shortlist_path.is_file():
                    payload = json.loads(shortlist_path.read_text(encoding="utf-8"))
                    by_id = {item.experiment_id: item for item in pipeline_grid()}
                    selected_pipelines = [
                        by_id[item["experiment_id"]]
                        for item in payload[: args.shortlist_size]
                        if item.get("experiment_id") in by_id
                    ]
                else:
                    selected_pipelines = []
            for pipeline in selected_pipelines:
                baseline = _extract_audit_metrics(suite_paths(args)["pipeline"] / pipeline.experiment_id)
                if baseline.get("map50_95") is None:
                    errors.append(f"FP32 baseline audit missing for {pipeline.experiment_id}; run --stage filtering first.")
    if args.stage in {"all", "deployment"}:
        needed: set[str] = set()
        if any(fmt.startswith("onnx") or fmt.startswith("tflite") or fmt == "ncnn" for fmt in args.deployment_formats):
            needed.update({"onnx", "onnxslim"})
        if any(fmt.startswith("onnx") for fmt in args.deployment_formats):
            needed.add("onnxruntime")
        if any(fmt.startswith("tflite") for fmt in args.deployment_formats):
            needed.update({"tensorflow", "onnx2tf", "ai_edge_litert"})
        if "ncnn" in args.deployment_formats:
            needed.add("ncnn")
        for package in sorted(needed):
            if importlib.util.find_spec(package) is None:
                errors.append(f"Deployment dependency {package} is missing in {sys.executable}.")
    calibration = _calibration_dataset_yaml(args)
    if calibration is not None and not calibration.is_file():
        errors.append(f"Calibration YAML is missing: {calibration}")
    return errors


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
        "preflight_errors": _preflight_errors(args),
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
    deployment_candidates = _deployment_pareto_candidates(
        deployment_rows, min_precision=args.min_precision, min_recall=args.min_recall,
    )
    factorial = factorial_effects(detector_rows)
    shortlist = select_deployment_shortlist(
        pipeline_rows, args.shortlist_size,
        min_precision=args.min_precision, min_recall=args.min_recall,
    )

    _write_csv(paths["aggregated"] / "detector_results.csv", detector_rows)
    _write_csv(paths["aggregated"] / "pipeline_results.csv", pipeline_rows)
    _write_csv(paths["aggregated"] / "deployment_results.csv", deployment_rows)
    (paths["aggregated"] / "deployment_candidates.json").write_text(
        json.dumps({
            "provisional": True,
            "reason": "Host throughput is not Android latency; benchmark these candidates on the target phone.",
            "inference_contract": {
                "image_size": args.imgsz,
                "confidence_threshold": args.conf,
                "nms_iou": args.iou,
                "dataset_yaml": str(_detector_eval_data(args)) if _detector_eval_data(args) else None,
                "tracker": "botsort",
                "filtering_and_tracking": "separate application code; not embedded in exported detector",
                "tensor_io": "inspect the chosen artifact and verify golden-image parity before Android integration",
            },
            "quality_leader": deployment_candidates[0] if deployment_candidates else None,
            "pareto_candidates": deployment_candidates,
        }, indent=2) + "\n", encoding="utf-8"
    )
    (paths["aggregated"] / "factorial_effects.json").write_text(json.dumps(factorial, indent=2) + "\n", encoding="utf-8")
    (paths["aggregated"] / "deployment_shortlist.json").write_text(json.dumps(shortlist, indent=2) + "\n", encoding="utf-8")
    summary = {
        "detectors": detector_rows,
        "pipelines": pipeline_rows,
        "deployment": deployment_rows,
        "deployment_candidates": deployment_candidates,
        "factorial_effects": factorial,
        "deployment_shortlist": shortlist,
    }
    (paths["aggregated"] / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_suite_plots(paths["plots"], detector_rows, pipeline_rows, deployment_rows, factorial)
    write_final_report(paths["reports"] / "final_report.md", args, detector_rows, pipeline_rows, deployment_rows, factorial, shortlist, deployment_candidates)
    return summary


def factorial_effects(rows: list[dict[str, Any]], metric: str = "map50_95") -> dict[str, Any]:
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
    interactions: dict[str, float] = {}
    names = tuple(factors)
    for selected in (
        (names[0], names[1]), (names[0], names[2]),
        (names[1], names[2]), names,
    ):
        positive = []
        negative = []
        for detector in DETECTOR_GRID:
            sign = 1
            for factor in selected:
                sign *= 1 if factors[factor][detector.experiment_id] else -1
            (positive if sign > 0 else negative).append(values[detector.experiment_id])
        interactions["_x_".join(selected)] = round(_mean(positive) - _mean(negative), 6)
    effects["main_effects"] = main
    effects["interaction_effects"] = interactions
    effects["interpretation"] = _factorial_interpretation(main)
    return effects


def select_deployment_shortlist(
    rows: list[dict[str, Any]],
    limit: int = 8,
    *,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
) -> list[dict[str, Any]]:
    """Select up to limit nondominated held-out pipelines for export."""
    eligible: list[dict[str, Any]] = []
    for row in rows:
        accuracy = _as_float(row.get("map50_95"))
        precision = _as_float(row.get("precision"))
        recall = _as_float(row.get("recall"))
        fps = _as_float(row.get("fps"))
        size = _as_float(row.get("model_size_mb"))
        if None in (accuracy, precision, recall, fps, size) or fps <= 0 or size <= 0:
            continue
        if precision < min_precision or recall < min_recall:
            continue
        eligible.append({
            "experiment_id": row["experiment_id"],
            "detector_id": row.get("detector_id", row["experiment_id"]),
            "filtering_enabled": row.get("filtering_enabled", False),
            "map50_95": accuracy,
            "precision": precision,
            "recall": recall,
            "fps": fps,
            "model_size_mb": size,
            "selection_rule": "Pareto: maximize mAP50-95 and measured FPS; minimize checkpoint MB; rank by mAP50-95",
        })

    efficient: list[dict[str, Any]] = []
    for row in eligible:
        dominated = any(
            other is not row
            and other["map50_95"] >= row["map50_95"]
            and other["fps"] >= row["fps"]
            and other["model_size_mb"] <= row["model_size_mb"]
            and (
                other["map50_95"] > row["map50_95"]
                or other["fps"] > row["fps"]
                or other["model_size_mb"] < row["model_size_mb"]
            )
            for other in eligible
        )
        if not dominated:
            efficient.append(row)
    efficient.sort(key=lambda row: (-row["map50_95"], -row["fps"], row["model_size_mb"], row["experiment_id"]))
    return efficient[:limit]



def _deployment_pareto_candidates(
    rows: list[dict[str, Any]], *, min_precision: float, min_recall: float,
) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "success"]
    ranked = select_deployment_shortlist(
        [
            {
                "experiment_id": row["deployment_id"],
                "detector_id": row["detector_id"],
                "filtering_enabled": row["filtering_enabled"],
                "map50_95": row.get("map50_95"),
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "fps": row.get("fps"),
                "model_size_mb": row.get("model_size_mb"),
            }
            for row in successful
        ],
        limit=len(successful),
        min_precision=min_precision,
        min_recall=min_recall,
    )
    by_id = {row["deployment_id"]: row for row in successful}
    return [by_id[item["experiment_id"]] for item in ranked]


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
    _bar_plot(plt, out / "false_positives_per_frame.png", detector_rows, "experiment_id", "pipeline_false_positives_per_frame", "Unfiltered Video False Positives per Frame")
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
    deployment_candidates: list[dict[str, Any]],
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
        "mAP, precision, recall and F1 use the held-out image test split. The FP/frame column uses the separate unfiltered held-out video; compare its values only within that video evaluation.",
        "",
        _markdown_table(detector_rows, ["experiment_id", "case", "map50", "map50_95", "precision", "recall", "f1", "pipeline_false_positives_per_frame", "status"]),
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
        _markdown_table(shortlist, ["experiment_id", "detector_id", "filtering_enabled", "map50_95", "precision", "recall", "fps", "model_size_mb"]),
        "",
        "## Provisional Android Candidates",
        "",
        "These candidates are Pareto efficient on held-out accuracy, host throughput and file size. The first row leads on mAP50-95 among candidates that meet the configured precision/recall floors. Benchmark on the Android device before choosing the final model.",
        "",
        _markdown_table(deployment_candidates, ["deployment_id", "map50_95", "precision", "recall", "fps", "model_size_mb", "exported_model"]),
        "",
        "## Deployment Benchmarks",
        "",
        "Host timing is end-to-end subprocess timing, including startup and audit work. Host RAM was not measured. Measure latency and peak memory again on the target Android device.",
        "",
        _markdown_table(deployment_rows, ["deployment_id", "detector_id", "filtering_enabled", "format", "map50", "map50_95", "delta_map50", "delta_map50_95", "precision", "recall", "delta_precision", "delta_recall", "fps", "latency_ms", "model_size_mb", "size_reduction_vs_pt_pct", "exported_model", "status"]),
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
    checkpoint = Path(record.metadata["checkpoint"])
    with log_path.open("w", encoding="utf-8") as log:
        exported_model = _exported_model_from_report(export_dir, fmt)
        if exported_model is None or exported_model.stat().st_mtime < checkpoint.stat().st_mtime:
            log.write("== export ==\n")
            log.flush()
            export_result = subprocess.run(
                record.command, cwd=repo_root(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
            if export_result.returncode != 0:
                record.status = "failed"
                record.returncode = int(export_result.returncode)
                record.error_message = f"Export failed for {fmt}; see {log_path}"
                return record
            exported_model = _exported_model_from_report(export_dir, fmt)
        if exported_model is None:
            record.status = "failed"
            record.returncode = 1
            record.error_message = f"No successful {fmt} model in {export_dir / 'export_report.json'}"
            return record
        benchmark_command = _deployment_audit_command(args, record, exported_model)
        record.metadata["exported_model"] = str(exported_model)
        record.metadata["benchmark_command"] = benchmark_command
        log.write("\n== held-out audit benchmark ==\n")
        log.flush()
        start = time.perf_counter()
        benchmark_result = subprocess.run(
            benchmark_command, cwd=repo_root(), env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
        duration = time.perf_counter() - start
        record.returncode = int(benchmark_result.returncode)
        if benchmark_result.returncode != 0:
            record.status = "failed"
            record.error_message = f"Deployment audit failed for {fmt}; see {log_path}"
            return record
        metrics = _extract_audit_metrics(Path(record.artifact_dir))
        frames = _as_float(metrics.get("frames_evaluated"))
        if metrics.get("map50_95") is None or not frames:
            record.status = "failed"
            record.returncode = 1
            record.error_message = f"Deployment audit produced no held-out detection metrics; see {log_path}"
            return record
        timing = {
            "benchmark_duration_s": round(duration, 4),
            "frames_evaluated": int(frames),
            "effective_fps": round(frames / duration, 4) if duration > 0 else None,
            "latency_ms": round(1000 * duration / frames, 4),
            "model_size_mb": _file_size_mb(exported_model),
            "exported_model": str(exported_model),
            "device": args.deployment_device,
            "timing_scope": "host end-to-end subprocess; includes startup, tracking, filtering and reporting",
        }
        (Path(record.artifact_dir) / "benchmark_timing.json").write_text(
            json.dumps(timing, indent=2) + "\n", encoding="utf-8"
        )
        record.status = "success"
        return record



def _write_stage_manifest(args: argparse.Namespace, record: StageRecord) -> Path:
    path = suite_paths(args)["manifests"] / f"{record.stage}_{record.experiment_id}.json"
    payload = {**asdict(record), "run_signature": _run_signature(args)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
        "--uniform-mask-safe-augmentations",
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
        str(args.detector_eval_conf),
        "--iou",
        str(args.detector_eval_iou),
        "--output-json",
        str(out / "metrics.json"),
    ]
    return StageRecord(detector.experiment_id, "detector_eval", command=cmd, artifact_dir=str(out))


def _deployment_record(args: argparse.Namespace, pipeline: PipelineExperiment, fmt: str) -> StageRecord:
    detector = detector_by_id(pipeline.detector_id)
    deployment_id = f"{pipeline.experiment_id}_{fmt.upper()}"
    export_dir = suite_paths(args)["exports"] / f"{detector.experiment_id}_{fmt.upper()}"
    bench_dir = suite_paths(args)["benchmarks"] / deployment_id
    checkpoint = _detector_artifact_dir(args, detector) / "best.pt"
    cmd = [
        sys.executable,
        "scripts/export_quantized_models.py",
        "--weights",
        str(checkpoint),
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
    return StageRecord(
        deployment_id,
        "deployment",
        command=cmd,
        artifact_dir=str(bench_dir),
        metadata={
            "pipeline_id": pipeline.experiment_id,
            "detector_id": detector.experiment_id,
            "filtering_enabled": pipeline.filtering_enabled,
            "format": fmt,
            "export_dir": str(export_dir),
            "checkpoint": str(checkpoint),
        },
    )


def _exported_model_from_report(export_dir: Path, fmt: str) -> Path | None:
    report_path = export_dir / "export_report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = next(
        (item for item in report.get("results", []) if item.get("format_name") == fmt and item.get("status") == "success"),
        None,
    )
    if row is None or not row.get("output_path"):
        return None
    path = Path(row["output_path"])
    if not path.is_absolute():
        path = _resolve(path)
    return path if path.exists() else None


def _deployment_audit_command(args: argparse.Namespace, record: StageRecord, exported_model: Path) -> list[str]:
    video = _pipeline_video(args)
    command = [
        sys.executable, "-m", "evaluation.eval_pres.run_audit",
        "--video", str(video),
        "--model", str(exported_model),
        "--output-dir", record.artifact_dir,
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--imgsz", str(args.imgsz),
        "--device", args.deployment_device,
        "--tracker", "botsort",
        "--vid-stride", str(args.vid_stride),
        "--gt-labels", str(_resolve(args.gt_labels)),
        "--no-save-video",
    ]
    if not record.metadata["filtering_enabled"]:
        command.append("--disable-multicue")
    return command



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
        "map50": evaluation.get("map50"),
        "map50_95": evaluation.get("map50_95"),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
        "training_val_map50_95": training.get("map50_95"),
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
    detector = detector_by_id(item.detector_id)
    stage_path = suite_paths(args)["manifests"] / f"filtering_{item.experiment_id}.json"
    duration: float | None = None
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage.get("status") == "success":
            duration = _as_float(stage.get("duration_s"))
    frames = _as_float(metrics.get("frames_evaluated"))
    fps = round(frames / duration, 4) if duration and frames else None
    return {
        "experiment_id": item.experiment_id,
        "detector_id": item.detector_id,
        "filtering_enabled": item.filtering_enabled,
        "status": "success" if metrics and metrics.get("map50_95") is not None else "missing",
        **metrics,
        "fps": fps,
        "model_size_mb": _file_size_mb(_detector_artifact_dir(args, detector) / "best.pt"),
    }



def _deployment_summary_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pipeline_by_id = {item.experiment_id: item for item in pipeline_grid()}
    for path in suite_paths(args)["benchmarks"].glob("*"):
        if not path.is_dir():
            continue
        pipeline_id, _, fmt = path.name.partition("_")
        pipeline = pipeline_by_id.get(pipeline_id)
        if pipeline is None or not fmt:
            continue
        metrics = _extract_deployment_metrics(path)
        baseline = _extract_audit_metrics(suite_paths(args)["pipeline"] / pipeline_id)
        exported_model = metrics.get("exported_model")
        complete = (
            metrics.get("map50_95") is not None
            and metrics.get("fps") is not None
            and bool(exported_model)
            and Path(exported_model).exists()
        )
        baseline_size = _file_size_mb(
            _detector_artifact_dir(args, detector_by_id(pipeline.detector_id)) / "best.pt"
        )
        deployed_size = _as_float(metrics.get("model_size_mb"))
        row = {
            "deployment_id": path.name,
            "pipeline_id": pipeline_id,
            "detector_id": pipeline.detector_id,
            "filtering_enabled": pipeline.filtering_enabled,
            "format": fmt.lower(),
            "status": "success" if complete else "missing",
            **metrics,
            "size_reduction_vs_pt_pct": (
                round(100 * (1 - deployed_size / baseline_size), 4)
                if deployed_size is not None and baseline_size else None
            ),
        }
        for key in ("map50", "map50_95", "precision", "recall", "f1"):
            deployed = _as_float(metrics.get(key))
            fp32 = _as_float(baseline.get(key))
            row[f"delta_{key}"] = round(deployed - fp32, 6) if deployed is not None and fp32 is not None else None
        rows.append(row)
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
    audit = _extract_audit_metrics(out)
    timing_path = out / "benchmark_timing.json"
    if not timing_path.is_file():
        return audit
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    return {
        **audit,
        "fps": timing.get("effective_fps"),
        "latency_ms": timing.get("latency_ms"),
        "model_size_mb": timing.get("model_size_mb"),
        "memory_usage_mb": timing.get("memory_usage_mb"),
        "exported_model": timing.get("exported_model"),
        "device": timing.get("device"),
        "timing_scope": timing.get("timing_scope"),
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
