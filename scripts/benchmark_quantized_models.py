#!/usr/bin/env python3
"""
Benchmark all quantized model variants through the full eval_pres pipeline.

Runs each exported model through demo-video to measure accuracy and latency,
then produces a side-by-side comparison table and JSON report.

Usage
-----
    # Auto-discover models from export directory:
    python scripts/benchmark_quantized_models.py \
        --export-dir runs/quantized_exports \
        --video /path/to/test_video.mp4 \
        --out runs/quantized_benchmarks

    # Also include the original .pt baseline:
    python scripts/benchmark_quantized_models.py \
        --export-dir runs/quantized_exports \
        --baseline /path/to/best.pt \
        --video /path/to/test_video.mp4 \
        --out runs/quantized_benchmarks

    # Manual model paths instead of auto-discovery:
    python scripts/benchmark_quantized_models.py \
        --models best.pt:baseline best.onnx:onnx_fp32 best_ncnn_model/:ncnn \
        --video /path/to/test_video.mp4 \
        --out runs/quantized_benchmarks

    # Pass extra flags through to demo-video:
    python scripts/benchmark_quantized_models.py \
        --export-dir runs/quantized_exports \
        --video /path/to/test_video.mp4 \
        --out runs/quantized_benchmarks \
        --preset quick --ground-truth /path/to/gt.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ── Model discovery ──────────────────────────────────────────────────────────

# File extensions / directory patterns that Ultralytics YOLO() can load
MODEL_GLOBS = {
    "onnx":        ["*.onnx"],
    "tflite":      ["*.tflite"],
    "torchscript": ["*.torchscript"],
    "engine":      ["*.engine"],
    "coreml":      ["*.mlpackage", "*.mlmodel"],
    "ncnn":        [],           # ncnn is a directory containing param + bin
    "pt":          ["*.pt"],
}


def _find_model_file(directory: Path) -> Path | None:
    """Find the loadable model file/directory inside an export subdirectory."""
    # For ncnn: the export directory itself (or a subdirectory) contains .param + .bin
    param_files = list(directory.rglob("*.param"))
    if param_files:
        # Return the parent directory of the .param file — that's what YOLO() loads
        return param_files[0].parent

    # For everything else: find the first matching model file
    for ext_list in MODEL_GLOBS.values():
        for pattern in ext_list:
            matches = list(directory.rglob(pattern))
            if matches:
                return matches[0]
    return None


def discover_models(export_dir: Path, baseline: Path | None = None) -> list[tuple[str, Path]]:
    """
    Auto-discover exported models from the directory structure created by
    export_quantized_models.py.

    Returns a list of (label, model_path) tuples.
    """
    models: list[tuple[str, Path]] = []

    # Add the baseline .pt model first if provided
    if baseline and baseline.exists():
        models.append(("baseline_pt", baseline))

    if not export_dir.is_dir():
        return models

    # Read export_report.json if available for ordering
    report_path = export_dir / "export_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        for result in report.get("results", []):
            if result.get("status") != "success":
                continue
            fmt_name = result["format_name"]
            output_path = Path(result["output_path"])
            resolved = None

            # 1. Try as absolute path
            if output_path.is_absolute() and output_path.exists():
                resolved = output_path
            # 2. Try relative to CWD
            elif output_path.exists():
                resolved = output_path.resolve()
            else:
                # 3. Try resolving relative path against export_dir's ancestors
                #    e.g. output_path="runs/quantized_exports/onnx_fp32/best.onnx"
                #    and export_dir="/abs/path/to/src/runs/quantized_exports"
                #    We walk up export_dir looking for a match
                for ancestor in [export_dir, export_dir.parent, export_dir.parent.parent,
                                 export_dir.parent.parent.parent]:
                    candidate = ancestor / output_path
                    if candidate.exists():
                        resolved = candidate
                        break
                    # Also try just the filename portion under export_dir/format_name/
                    subdir_candidate = export_dir / fmt_name
                    if subdir_candidate.is_dir():
                        found = _find_model_file(subdir_candidate)
                        if found:
                            resolved = found
                            break

            if resolved:
                models.append((fmt_name, resolved))
            else:
                # Last resort: scan the format subdirectory directly
                subdir = export_dir / fmt_name
                if subdir.is_dir():
                    found = _find_model_file(subdir)
                    if found:
                        models.append((fmt_name, found))
                    else:
                        print(f"  WARNING: Could not find model for {fmt_name} in {subdir}", file=sys.stderr)
                else:
                    print(f"  WARNING: Could not resolve model path for {fmt_name}: {output_path}", file=sys.stderr)
    else:
        # Fallback: scan subdirectories
        for subdir in sorted(export_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            model_file = _find_model_file(subdir)
            if model_file:
                models.append((subdir.name, model_file))

    return models


def parse_manual_models(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse --models path:label specifications."""
    models = []
    for spec in specs:
        if ":" in spec:
            path_str, label = spec.rsplit(":", 1)
        else:
            path_str = spec
            label = Path(spec).stem
        models.append((label, Path(path_str)))
    return models


# ── Single benchmark run ─────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    label: str
    model_path: str
    status: str                                  # "success" | "error"
    error_message: str = ""

    # File info
    model_size_mb: float = 0.0

    # Latency — pre-measurement (from the timing block we added)
    num_frames: int = 0
    detection_total_s: float = 0.0
    tracking_total_s: float = 0.0
    filtering_total_s: float = 0.0
    pre_measurement_total_s: float = 0.0
    per_frame_avg_detection_ms: float = 0.0
    per_frame_avg_tracking_ms: float = 0.0
    per_frame_avg_filtering_ms: float = 0.0
    per_frame_avg_total_ms: float = 0.0
    effective_fps: float = 0.0

    # Full pipeline
    pipeline_elapsed_s: float = 0.0

    # Detection counts
    total_detections: int = 0
    linked_tracks: int = 0
    measurement_tracks: int = 0
    measurement_reports: int = 0

    # Evaluation metrics (subset of key ones)
    computed_metrics: int = 0
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)

    # Per-frame detection latencies
    per_frame_detection_ms: list[float] = field(default_factory=list)

    # Accuracy vs baseline (populated after all runs)
    accuracy: dict[str, Any] = field(default_factory=dict)


def _model_size_mb(path: Path) -> float:
    """Compute total size of a model file or directory in MB."""
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 2)
    return 0.0


def run_single_benchmark(
    label: str,
    model_path: Path,
    video: Path,
    out_dir: Path,
    demo_kwargs: dict[str, Any],
) -> BenchmarkResult:
    """Run one model through the full demo-video pipeline and collect results."""

    # Add PYTHONPATH so imports work
    src_eval = Path(__file__).resolve().parent.parent / "src" / "evaluation"
    if str(src_eval) not in sys.path:
        sys.path.insert(0, str(src_eval))

    from eval_pres.video_demo import run_video_demo

    result = BenchmarkResult(
        label=label,
        model_path=str(model_path),
        status="error",
        model_size_mb=_model_size_mb(model_path),
    )

    run_out = out_dir / label
    try:
        summary = run_video_demo(
            video=video,
            out=run_out,
            model_path=model_path,
            **demo_kwargs,
        )
    except Exception as exc:
        result.error_message = str(exc)
        return result

    result.status = "success"
    result.pipeline_elapsed_s = summary.get("elapsed_seconds", 0.0)
    result.total_detections = summary.get("detections", 0)
    result.linked_tracks = summary.get("linked_tracks", 0)
    result.measurement_tracks = summary.get("measurement_tracks_used", 0)
    result.measurement_reports = summary.get("measurement_reports", 0)
    result.computed_metrics = summary.get("evaluation_computed_metrics", 0)

    # Extract pre-measurement timing
    timing = summary.get("pre_measurement_timing", {})
    if timing:
        result.num_frames = timing.get("num_frames", 0)
        result.detection_total_s = timing.get("detection_total_s", 0.0)
        result.tracking_total_s = timing.get("tracking_total_s", 0.0)
        result.filtering_total_s = timing.get("filtering_total_s", 0.0)
        result.pre_measurement_total_s = timing.get("pre_measurement_total_s", 0.0)
        result.per_frame_avg_detection_ms = timing.get("per_frame_avg_detection_ms", 0.0)
        result.per_frame_avg_tracking_ms = timing.get("per_frame_avg_tracking_ms", 0.0)
        result.per_frame_avg_filtering_ms = timing.get("per_frame_avg_filtering_ms", 0.0)
        result.per_frame_avg_total_ms = timing.get("per_frame_avg_total_ms", 0.0)
        result.per_frame_detection_ms = timing.get("per_frame_detection_ms", [])
        if result.per_frame_avg_total_ms > 0:
            result.effective_fps = round(1000.0 / result.per_frame_avg_total_ms, 2)

    # Extract evaluation metrics from the evaluation summary
    eval_summary_path = run_out / "evaluation" / "evaluation_summary.json"
    if eval_summary_path.exists():
        try:
            with open(eval_summary_path) as f:
                eval_data = json.load(f)
            for metric in eval_data.get("metrics", []):
                if metric.get("status") == "computed" and metric.get("value") is not None:
                    key = f"{metric['section']}/{metric['metric']}"
                    result.evaluation_metrics[key] = {
                        "value": metric["value"],
                        "unit": metric.get("unit", ""),
                        "direction": metric.get("direction", ""),
                    }
        except Exception:
            pass

    return result


# ── Accuracy comparison against baseline ─────────────────────────────────────

def _bbox_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_manifest_detections(manifest_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load clip_manifest.json and return detections grouped by frame_id."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for track in manifest.get("tracks", []):
        fid = int(track.get("frame_id", 0))
        by_frame.setdefault(fid, []).append(track)
    return by_frame


def _load_measurement_statuses(reports_path: Path) -> dict[str, str]:
    """Load reports.json and return {track_id: overall_category}."""
    if not reports_path.exists():
        return {}
    with open(reports_path) as f:
        reports = json.load(f)
    statuses: dict[str, str] = {}
    for report in reports:
        # Handle both dict and object-style reports
        if isinstance(report, dict):
            tid = str(report.get("lamp_track_id", ""))
            metrics = report.get("metrics", {}) or {}
            cat = str(metrics.get("overall_category", "unknown"))
            if tid:
                statuses[tid] = cat
    return statuses


@dataclass
class AccuracyComparison:
    label: str
    detection_match_pct: float = 0.0      # % of baseline dets matched
    mean_iou_matched: float = 0.0         # mean IoU of matched pairs
    missed_pct: float = 0.0               # % of baseline dets NOT found
    extra_pct: float = 0.0                # extra dets as % of baseline count
    score_correlation: float = 0.0        # Pearson correlation of matched scores
    mean_score_delta: float = 0.0         # avg(model_score - baseline_score)
    status_agreement_pct: float = 0.0     # % of tracks with same measurement category
    total_baseline_dets: int = 0
    total_model_dets: int = 0
    total_matched: int = 0
    total_missed: int = 0
    total_extra: int = 0
    status_compared: int = 0
    status_agreed: int = 0


def compare_accuracy_against_baseline(
    baseline_result: BenchmarkResult,
    other_results: list[BenchmarkResult],
    out_dir: Path,
    iou_threshold: float = 0.50,
) -> list[AccuracyComparison]:
    """Compare each model's detections against the baseline using IoU matching."""
    # run_video_demo resolves relative paths via repo_root(), so we must too
    resolved_out = out_dir
    if not out_dir.is_absolute():
        try:
            from eval_pres.artifact_utils import repo_root
            resolved_out = repo_root() / out_dir
        except ImportError:
            resolved_out = out_dir

    # Try resolved path first, fall back to raw out_dir
    baseline_dir = resolved_out / baseline_result.label
    if not (baseline_dir / "clip_manifest.json").exists():
        baseline_dir = out_dir / baseline_result.label

    baseline_manifest = baseline_dir / "clip_manifest.json"
    baseline_reports = baseline_dir / "measurement" / "reports.json"

    if not baseline_manifest.exists():
        print(f"  WARNING: Baseline clip_manifest.json not found at {baseline_manifest}, "
              f"skipping accuracy comparison.",
              file=sys.stderr)
        return []

    base_dets = _load_manifest_detections(baseline_manifest)
    base_statuses = _load_measurement_statuses(baseline_reports)

    comparisons: list[AccuracyComparison] = []

    for result in other_results:
        model_dir = resolved_out / result.label
        if not (model_dir / "clip_manifest.json").exists():
            model_dir = out_dir / result.label
        model_manifest = model_dir / "clip_manifest.json"
        model_reports = model_dir / "measurement" / "reports.json"

        comp = AccuracyComparison(label=result.label)

        if not model_manifest.exists():
            comparisons.append(comp)
            continue

        model_dets = _load_manifest_detections(model_manifest)
        model_statuses = _load_measurement_statuses(model_reports)

        # Collect all frame IDs
        all_frames = sorted(set(base_dets.keys()) | set(model_dets.keys()))

        total_base = 0
        total_model = 0
        total_matched = 0
        matched_ious: list[float] = []
        matched_base_scores: list[float] = []
        matched_model_scores: list[float] = []

        for fid in all_frames:
            b_list = base_dets.get(fid, [])
            m_list = list(model_dets.get(fid, []))  # copy so we can pop
            total_base += len(b_list)
            total_model += len(m_list)

            used_model: set[int] = set()
            for b_det in b_list:
                b_bbox = b_det.get("bbox_xyxy", [0, 0, 0, 0])
                b_score = float(b_det.get("detector_score", 0.0))
                best_idx = -1
                best_iou = 0.0
                for idx, m_det in enumerate(m_list):
                    if idx in used_model:
                        continue
                    iou = _bbox_iou(b_bbox, m_det.get("bbox_xyxy", [0, 0, 0, 0]))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx >= 0 and best_iou >= iou_threshold:
                    total_matched += 1
                    matched_ious.append(best_iou)
                    matched_base_scores.append(b_score)
                    matched_model_scores.append(
                        float(m_list[best_idx].get("detector_score", 0.0))
                    )
                    used_model.add(best_idx)

        comp.total_baseline_dets = total_base
        comp.total_model_dets = total_model
        comp.total_matched = total_matched
        comp.total_missed = total_base - total_matched
        comp.total_extra = max(0, total_model - total_matched)

        if total_base > 0:
            comp.detection_match_pct = round(total_matched / total_base * 100, 2)
            comp.missed_pct = round(comp.total_missed / total_base * 100, 2)
            comp.extra_pct = round(comp.total_extra / total_base * 100, 2)
        if matched_ious:
            comp.mean_iou_matched = round(sum(matched_ious) / len(matched_ious), 4)
        if matched_base_scores and matched_model_scores:
            comp.mean_score_delta = round(
                sum(m - b for m, b in zip(matched_model_scores, matched_base_scores))
                / len(matched_base_scores), 4
            )
            # Pearson correlation
            n = len(matched_base_scores)
            if n > 1:
                mean_b = sum(matched_base_scores) / n
                mean_m = sum(matched_model_scores) / n
                cov = sum((b - mean_b) * (m - mean_m)
                          for b, m in zip(matched_base_scores, matched_model_scores)) / n
                std_b = math.sqrt(sum((b - mean_b) ** 2 for b in matched_base_scores) / n)
                std_m = math.sqrt(sum((m - mean_m) ** 2 for m in matched_model_scores) / n)
                if std_b > 0 and std_m > 0:
                    comp.score_correlation = round(cov / (std_b * std_m), 4)
                else:
                    comp.score_correlation = 1.0  # constant scores = perfect agreement

        # Measurement status agreement
        common_tracks = set(base_statuses.keys()) & set(model_statuses.keys())
        comp.status_compared = len(common_tracks)
        comp.status_agreed = sum(
            1 for tid in common_tracks if base_statuses[tid] == model_statuses[tid]
        )
        if comp.status_compared > 0:
            comp.status_agreement_pct = round(
                comp.status_agreed / comp.status_compared * 100, 2
            )

        comparisons.append(comp)

    return comparisons


def _print_accuracy_table(
    baseline_label: str,
    comparisons: list[AccuracyComparison],
) -> None:
    print("\n" + "=" * 115)
    print("  ACCURACY COMPARISON vs BASELINE  (baseline = " + baseline_label + ")")
    print("=" * 115)
    header = (
        f"  {'Model':<20} {'Match%':>8} {'MeanIoU':>9} {'Missed%':>9} {'Extra%':>8}"
        f" {'ScoreCorr':>10} {'Δ Score':>9} {'StatusAgree%':>13}"
    )
    print(header)
    print("  " + "─" * 111)

    # Baseline row (self-comparison)
    print(
        f"  {baseline_label:<20} {'100.00':>8} {'1.0000':>9} {'0.00':>9} {'0.00':>8}"
        f" {'1.0000':>10} {'+0.0000':>9} {'100.00':>13}"
    )

    for c in comparisons:
        delta_str = f"{c.mean_score_delta:>+.4f}" if c.total_matched else "—"
        corr_str = f"{c.score_correlation:.4f}" if c.total_matched else "—"
        print(
            f"  {c.label:<20} {c.detection_match_pct:>8.2f} {c.mean_iou_matched:>9.4f}"
            f" {c.missed_pct:>9.2f} {c.extra_pct:>8.2f}"
            f" {corr_str:>10} {delta_str:>9} {c.status_agreement_pct:>13.2f}"
        )

    print("=" * 115)

    # Detailed breakdown
    print("\n  DETAILED COUNTS:")
    print("  " + "─" * 80)
    print(
        f"  {'Model':<20} {'Base Dets':>10} {'Model Dets':>11} {'Matched':>9}"
        f" {'Missed':>8} {'Extra':>7} {'Status Cmp':>11} {'Agreed':>8}"
    )
    print("  " + "─" * 80)
    for c in comparisons:
        print(
            f"  {c.label:<20} {c.total_baseline_dets:>10} {c.total_model_dets:>11}"
            f" {c.total_matched:>9} {c.total_missed:>8} {c.total_extra:>7}"
            f" {c.status_compared:>11} {c.status_agreed:>8}"
        )
    print()


# ── Main benchmark runner ────────────────────────────────────────────────────

@dataclass
class BenchmarkReport:
    video: str
    timestamp: str
    demo_kwargs: dict[str, Any]
    results: list[BenchmarkResult] = field(default_factory=list)


def run_benchmarks(
    models: list[tuple[str, Path]],
    video: Path,
    out: Path,
    demo_kwargs: dict[str, Any],
) -> BenchmarkReport:
    out.mkdir(parents=True, exist_ok=True)

    report = BenchmarkReport(
        video=str(video),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        demo_kwargs={k: str(v) if isinstance(v, Path) else v for k, v in demo_kwargs.items()},
    )

    n = len(models)
    print("\n" + "=" * 78)
    print("  QUANTIZED MODEL BENCHMARK — FULL EVALUATION STACK")
    print("=" * 78)
    print(f"  Video          : {video}")
    print(f"  Output         : {out}")
    print(f"  Models to test : {n}")
    for i, (label, path) in enumerate(models, 1):
        print(f"    [{i}] {label:<20} → {path}")
    print("=" * 78 + "\n")

    for i, (label, model_path) in enumerate(models, 1):
        print(f"\n{'─' * 78}")
        print(f"  [{i}/{n}] Running benchmark: {label}")
        print(f"  Model: {model_path}  ({_model_size_mb(model_path):.2f} MB)")
        print(f"{'─' * 78}")

        if not model_path.exists():
            print(f"  ✗ SKIPPED — model path does not exist")
            report.results.append(BenchmarkResult(
                label=label,
                model_path=str(model_path),
                status="error",
                error_message="Model path does not exist",
            ))
            continue

        t0 = time.perf_counter()
        result = run_single_benchmark(label, model_path, video, out, demo_kwargs)
        wall_time = time.perf_counter() - t0
        report.results.append(result)

        if result.status == "success":
            print(f"  ✓ Completed in {wall_time:.1f}s")
            print(f"    Detections: {result.total_detections}  |  Tracks: {result.linked_tracks}  |  Reports: {result.measurement_reports}")
            print(f"    Per-frame detection: {result.per_frame_avg_detection_ms:.2f} ms  |  Total pre-meas: {result.per_frame_avg_total_ms:.2f} ms/frame  |  FPS: {result.effective_fps:.2f}")
        else:
            print(f"  ✗ FAILED — {result.error_message}")

    # ── Comparison table ─────────────────────────────────────────────────
    succeeded = [r for r in report.results if r.status == "success"]
    accuracy_comparisons: list[AccuracyComparison] = []
    if succeeded:
        _print_comparison_table(succeeded)

        # ── Accuracy comparison vs baseline ──────────────────────────────
        if len(succeeded) > 1:
            baseline = succeeded[0]
            others = succeeded[1:]
            accuracy_comparisons = compare_accuracy_against_baseline(
                baseline, others, out
            )
            if accuracy_comparisons:
                _print_accuracy_table(baseline.label, accuracy_comparisons)
                # Attach accuracy data to each result for JSON/CSV
                for comp in accuracy_comparisons:
                    for r in succeeded:
                        if r.label == comp.label:
                            r.accuracy = asdict(comp)
                            break

        _write_comparison_csv(out / "benchmark_comparison.csv", succeeded,
                              accuracy_comparisons)

    # Save full report
    report_path = out / "benchmark_report.json"

    def _serialize(obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            return {k: _serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [_serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    with open(report_path, "w") as f:
        json.dump(_serialize(report), f, indent=2)
    print(f"\n  Full report: {report_path}")
    print(f"  CSV table:   {out / 'benchmark_comparison.csv'}\n")

    # Save accuracy comparison as separate JSON for easy access
    if accuracy_comparisons:
        acc_path = out / "accuracy_comparison.json"
        with open(acc_path, "w") as f:
            json.dump([asdict(c) for c in accuracy_comparisons], f, indent=2)
        print(f"  Accuracy:    {acc_path}\n")

    return report


def _print_comparison_table(results: list[BenchmarkResult]) -> None:
    print("\n" + "=" * 110)
    print("  BENCHMARK COMPARISON TABLE")
    print("=" * 110)

    header = (
        f"  {'Model':<20} {'Size(MB)':>9} {'Det(ms)':>9} {'Track(ms)':>10} "
        f"{'Filter(ms)':>11} {'Total(ms)':>10} {'FPS':>7} {'Dets':>6} {'Tracks':>7} {'Reports':>8}"
    )
    print(header)
    print("  " + "─" * 106)

    # Find baseline (first result) for relative comparison
    baseline = results[0]

    for r in results:
        speedup = ""
        if r is not baseline and baseline.per_frame_avg_total_ms > 0 and r.per_frame_avg_total_ms > 0:
            ratio = baseline.per_frame_avg_total_ms / r.per_frame_avg_total_ms
            speedup = f" ({ratio:.2f}×)" if ratio >= 1.0 else f" ({ratio:.2f}×)"

        row = (
            f"  {r.label:<20} {r.model_size_mb:>9.2f} {r.per_frame_avg_detection_ms:>9.2f} "
            f"{r.per_frame_avg_tracking_ms:>10.2f} {r.per_frame_avg_filtering_ms:>11.2f} "
            f"{r.per_frame_avg_total_ms:>10.2f} {r.effective_fps:>7.2f} "
            f"{r.total_detections:>6} {r.linked_tracks:>7} {r.measurement_reports:>8}"
            f"{speedup}"
        )
        print(row)

    print("=" * 110)

    # Accuracy drift check
    if len(results) > 1:
        print("\n  DETECTION CONSISTENCY (vs. first model):")
        print("  " + "─" * 60)
        base_dets = baseline.total_detections
        base_tracks = baseline.linked_tracks
        for r in results[1:]:
            det_diff = r.total_detections - base_dets
            track_diff = r.linked_tracks - base_tracks
            det_pct = (det_diff / base_dets * 100) if base_dets else 0
            track_pct = (track_diff / base_tracks * 100) if base_tracks else 0
            print(
                f"  {r.label:<20}  detections: {det_diff:>+5} ({det_pct:>+6.1f}%)  "
                f"tracks: {track_diff:>+4} ({track_pct:>+6.1f}%)"
            )

    # Evaluation metrics comparison (if available)
    all_metric_keys: set[str] = set()
    for r in results:
        all_metric_keys.update(r.evaluation_metrics.keys())

    if all_metric_keys:
        print(f"\n  EVALUATION METRICS COMPARISON:")
        print("  " + "─" * 90)
        sorted_keys = sorted(all_metric_keys)
        # Header
        metric_header = f"  {'Metric':<45}"
        for r in results:
            metric_header += f" {r.label:>12}"
        print(metric_header)
        print("  " + "─" * 90)
        for key in sorted_keys:
            row = f"  {key:<45}"
            for r in results:
                m = r.evaluation_metrics.get(key)
                if m is not None:
                    val = m["value"]
                    if isinstance(val, float):
                        row += f" {val:>12.4f}"
                    else:
                        row += f" {str(val):>12}"
                else:
                    row += f" {'—':>12}"
            print(row)
        print()


def _write_comparison_csv(
    path: Path,
    results: list[BenchmarkResult],
    accuracy_comparisons: list[AccuracyComparison] | None = None,
) -> None:
    """Write the comparison table as CSV for easy import into spreadsheets."""
    # Collect all metric keys
    all_metric_keys: list[str] = sorted({
        k for r in results for k in r.evaluation_metrics
    })

    accuracy_fields = [
        "acc_match_pct", "acc_mean_iou", "acc_missed_pct", "acc_extra_pct",
        "acc_score_corr", "acc_score_delta", "acc_status_agree_pct",
    ]

    fieldnames = [
        "model", "model_path", "size_mb",
        "num_frames",
        "det_total_s", "track_total_s", "filter_total_s", "pre_meas_total_s",
        "det_ms_per_frame", "track_ms_per_frame", "filter_ms_per_frame", "total_ms_per_frame",
        "effective_fps",
        "pipeline_elapsed_s",
        "detections", "linked_tracks", "measurement_tracks", "measurement_reports",
        "computed_metrics",
    ] + accuracy_fields + [f"metric:{k}" for k in all_metric_keys]

    # Build lookup for accuracy data
    acc_by_label: dict[str, AccuracyComparison] = {}
    if accuracy_comparisons:
        for c in accuracy_comparisons:
            acc_by_label[c.label] = c

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            acc = acc_by_label.get(r.label)
            row: dict[str, Any] = {
                "model": r.label,
                "model_path": r.model_path,
                "size_mb": r.model_size_mb,
                "num_frames": r.num_frames,
                "det_total_s": r.detection_total_s,
                "track_total_s": r.tracking_total_s,
                "filter_total_s": r.filtering_total_s,
                "pre_meas_total_s": r.pre_measurement_total_s,
                "det_ms_per_frame": r.per_frame_avg_detection_ms,
                "track_ms_per_frame": r.per_frame_avg_tracking_ms,
                "filter_ms_per_frame": r.per_frame_avg_filtering_ms,
                "total_ms_per_frame": r.per_frame_avg_total_ms,
                "effective_fps": r.effective_fps,
                "pipeline_elapsed_s": r.pipeline_elapsed_s,
                "detections": r.total_detections,
                "linked_tracks": r.linked_tracks,
                "measurement_tracks": r.measurement_tracks,
                "measurement_reports": r.measurement_reports,
                "computed_metrics": r.computed_metrics,
                "acc_match_pct": acc.detection_match_pct if acc else (100.0 if r == results[0] else ""),
                "acc_mean_iou": acc.mean_iou_matched if acc else (1.0 if r == results[0] else ""),
                "acc_missed_pct": acc.missed_pct if acc else (0.0 if r == results[0] else ""),
                "acc_extra_pct": acc.extra_pct if acc else (0.0 if r == results[0] else ""),
                "acc_score_corr": acc.score_correlation if acc else (1.0 if r == results[0] else ""),
                "acc_score_delta": acc.mean_score_delta if acc else (0.0 if r == results[0] else ""),
                "acc_status_agree_pct": acc.status_agreement_pct if acc else (100.0 if r == results[0] else ""),
            }
            for key in all_metric_keys:
                m = r.evaluation_metrics.get(key)
                row[f"metric:{key}"] = m["value"] if m else ""
            writer.writerow(row)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark quantized model variants through the full evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection (mutually exclusive groups)
    model_group = parser.add_argument_group("Model selection (pick one)")
    model_group.add_argument(
        "--export-dir",
        help="Directory from export_quantized_models.py. Auto-discovers all exported models.",
    )
    model_group.add_argument(
        "--baseline",
        help="Path to original .pt weights to include as the baseline reference.",
    )
    model_group.add_argument(
        "--models", nargs="+",
        help="Manual model specs as 'path:label' (e.g. 'best.onnx:onnx_fp32'). "
             "Overrides --export-dir.",
    )

    # Required
    parser.add_argument("--video", default="", help="Test video path (not needed with --compare-only).")
    parser.add_argument("--out", required=True, help="Benchmark output directory.")

    # Compare-only mode
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip benchmarks; just run accuracy comparison on existing outputs. "
                             "Expects --out to contain subdirectories with clip_manifest.json files.")
    parser.add_argument("--baseline-label", default="baseline_pt",
                        help="Label of the baseline model subdirectory (default: baseline_pt). "
                             "Used with --compare-only.")

    # demo-video pass-through options
    demo_group = parser.add_argument_group("demo-video options (passed through)")
    demo_group.add_argument("--preset", choices=["quick", "standard", "full"], default="quick",
                            help="Preset for demo-video (default: quick for faster benchmarking).")
    demo_group.add_argument("--fps-sample", type=float, default=3.0)
    demo_group.add_argument("--conf", type=float, default=0.05)
    demo_group.add_argument("--max-frames", type=int, help="Limit frames for faster benchmarking.")
    demo_group.add_argument("--max-det", type=int, default=12)
    demo_group.add_argument("--iou-link-threshold", type=float, default=0.30)
    demo_group.add_argument("--ground-truth", help="Ground truth JSON for accuracy metrics.")
    demo_group.add_argument("--route-distance-km", type=float)
    demo_group.add_argument("--measurement-max-tracks", type=int, default=30)

    return parser.parse_args()


def run_compare_only(out_dir: Path, baseline_label: str) -> None:
    """Run accuracy comparison on existing benchmark outputs without re-running inference."""
    # Add PYTHONPATH so imports work
    src_eval = Path(__file__).resolve().parent.parent / "src" / "evaluation"
    if str(src_eval) not in sys.path:
        sys.path.insert(0, str(src_eval))

    # Resolve the output directory the same way run_video_demo does
    resolved_out = out_dir
    if not out_dir.is_absolute():
        try:
            from eval_pres.artifact_utils import repo_root
            resolved_out = repo_root() / out_dir
        except ImportError:
            pass
    if not resolved_out.is_dir():
        resolved_out = out_dir

    # Discover subdirectories that contain clip_manifest.json
    labels: list[str] = []
    for subdir in sorted(resolved_out.iterdir()):
        if subdir.is_dir() and (subdir / "clip_manifest.json").exists():
            labels.append(subdir.name)

    if not labels:
        print(f"ERROR: No subdirectories with clip_manifest.json found in {resolved_out}",
              file=sys.stderr)
        sys.exit(1)

    if baseline_label not in labels:
        print(f"ERROR: Baseline '{baseline_label}' not found. Available: {labels}",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n  Compare-only mode: {resolved_out}")
    print(f"  Baseline: {baseline_label}")
    print(f"  Models:   {', '.join(labels)}\n")

    baseline_result = BenchmarkResult(label=baseline_label, model_path="", status="success")
    other_results = [
        BenchmarkResult(label=lbl, model_path="", status="success")
        for lbl in labels if lbl != baseline_label
    ]

    comparisons = compare_accuracy_against_baseline(
        baseline_result, other_results, out_dir
    )
    if comparisons:
        _print_accuracy_table(baseline_label, comparisons)
        acc_path = resolved_out / "accuracy_comparison.json"
        with open(acc_path, "w") as f:
            json.dump([asdict(c) for c in comparisons], f, indent=2)
        print(f"  Saved to: {acc_path}\n")
    else:
        print("  No comparisons produced.")


def main() -> None:
    args = parse_args()
    out = Path(args.out)

    if args.compare_only:
        run_compare_only(out, args.baseline_label)
        return

    video = Path(args.video)
    if not args.video:
        print("ERROR: --video is required unless using --compare-only.", file=sys.stderr)
        sys.exit(1)

    # Discover / parse models
    if args.models:
        models = parse_manual_models(args.models)
    elif args.export_dir:
        baseline = Path(args.baseline) if args.baseline else None
        models = discover_models(Path(args.export_dir), baseline=baseline)
    elif args.baseline:
        models = [("baseline_pt", Path(args.baseline))]
    else:
        print("ERROR: Provide --export-dir, --baseline, or --models.", file=sys.stderr)
        sys.exit(1)

    if not models:
        print("ERROR: No models found to benchmark.", file=sys.stderr)
        sys.exit(1)

    # Build demo-video kwargs (exclude model_path, video, out — those are set per-run)
    demo_kwargs: dict[str, Any] = {
        "fps_sample": args.fps_sample,
        "conf": args.conf,
        "max_frames": args.max_frames,
        "max_det": args.max_det,
        "iou_link_threshold": args.iou_link_threshold,
        "ground_truth": args.ground_truth,
        "route_distance_km": args.route_distance_km,
        "measurement_max_tracks": args.measurement_max_tracks,
        "preset": args.preset,
    }

    run_benchmarks(models, video, out, demo_kwargs)


if __name__ == "__main__":
    main()
