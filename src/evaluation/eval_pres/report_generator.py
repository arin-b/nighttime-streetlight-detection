"""Audit report generation (PDF §6 — output section).

Produces three output formats:
  1. JSON   — complete per-lamp data + metrics
  2. CSV    — flat table for spreadsheet analysis
  3. Markdown — human-readable summary report
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.eval_pres.aggregator import AggregatedLamp


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct(numerator: float, denominator: float) -> float:
    return round((numerator / denominator * 100.0), 2) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(len(ordered) - 1, idx))])


def _lamp_quality_summary(
    lamps: list[AggregatedLamp],
    filter_stats: dict[str, Any],
) -> dict[str, Any]:
    confidences = [l.confidence for l in lamps]
    frame_counts = [float(l.frame_count) for l in lamps]
    on_fractions = [
        l.frames_on / max(l.frames_on + l.frames_off, 1)
        for l in lamps
    ]
    total = len(lamps)
    working = sum(1 for l in lamps if l.status == "working")
    off = sum(1 for l in lamps if l.status == "off")
    flickering = sum(1 for l in lamps if l.status == "flickering")
    raw = int(filter_stats.get("total_raw_detections", 0) or 0)
    kept = int(filter_stats.get("total_kept_detections", 0) or 0)
    model_total = int(filter_stats.get("total_model_detections", raw) or 0)

    return {
        "working_rate_pct": _pct(working, total),
        "off_rate_pct": _pct(off, total),
        "flickering_rate_pct": _pct(flickering, total),
        "mean_detector_confidence": round(_mean(confidences), 4),
        "median_detector_confidence": round(_median(confidences), 4),
        "low_confidence_lamps_lt_0_5": sum(1 for l in lamps if l.confidence < 0.5),
        "median_track_frames": round(_median(frame_counts), 2),
        "min_track_frames": int(min(frame_counts)) if frame_counts else 0,
        "max_track_frames": int(max(frame_counts)) if frame_counts else 0,
        "mean_on_fraction_pct": round(_mean(on_fractions) * 100.0, 2),
        "target_class_acceptance_rate_pct": _pct(kept, raw),
        "target_detections_per_final_lamp": round(kept / total, 2) if total else 0.0,
        "non_target_suppression_rate_pct": _pct(
            int(filter_stats.get("non_target_detections_suppressed", 0) or 0),
            model_total,
        ),
    }


def _resolved_class_text(config: dict[str, Any]) -> str:
    resolved = config.get("detector", {}).get("resolved_class_names", {})
    if not resolved:
        return "not resolved"
    return ", ".join(
        f"{class_id}: {name}"
        for class_id, name in sorted(resolved.items(), key=lambda item: int(item[0]))
    )


# ------------------------------------------------------------------ #
# JSON report                                                        #
# ------------------------------------------------------------------ #

def write_json_report(
    output_dir: Path,
    lamps: list[AggregatedLamp],
    config: dict[str, Any],
    video_meta: dict[str, Any],
    filter_stats: dict[str, Any],
    eval_metrics: dict[str, Any] | None = None,
    location_prior_report: dict[str, Any] | None = None,
) -> Path:
    """Write the complete audit report as JSON."""
    quality_summary = _lamp_quality_summary(lamps, filter_stats)
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": config,
        "video": video_meta,
        "summary": {
            "total_lamps_detected": len(lamps),
            "lamps_working": sum(1 for l in lamps if l.status == "working"),
            "lamps_off": sum(1 for l in lamps if l.status == "off"),
            "lamps_flickering": sum(1 for l in lamps if l.status == "flickering"),
            "working_rate_pct": quality_summary["working_rate_pct"],
        },
        "quality_summary": quality_summary,
        "filter_statistics": filter_stats,
        "per_lamp_data": [l.to_dict() for l in lamps],
    }
    if eval_metrics:
        report["evaluation_metrics"] = eval_metrics
    else:
        report["evaluation_metrics"] = {
            "available": False,
            "reason": "No ground-truth labels were provided, so precision, recall, F1, and mAP were not computed.",
        }
    if location_prior_report:
        report["location_prior"] = location_prior_report

    path = output_dir / "audit_report.json"
    _write_json(path, report)
    return path


# ------------------------------------------------------------------ #
# CSV report                                                          #
# ------------------------------------------------------------------ #

def write_csv_report(
    output_dir: Path,
    lamps: list[AggregatedLamp],
) -> Path:
    """Write flat per-lamp CSV."""
    rows = []
    for lamp in lamps:
        observed_frames = max(lamp.frames_on + lamp.frames_off, 1)
        location = lamp.location or {}
        prior_match = (lamp.existence_prior or {}).get("match") or {}
        rows.append({
            "track_id": lamp.track_id,
            "status": lamp.status,
            "avg_brightness": round(lamp.avg_brightness, 2),
            "median_brightness": round(lamp.median_brightness, 2),
            "brightness_std": round(lamp.brightness_std, 2),
            "min_brightness": round(lamp.min_brightness, 2),
            "max_brightness": round(lamp.max_brightness, 2),
            "peak_brightness_mean": round(lamp.peak_brightness_mean, 2),
            "confidence": round(lamp.confidence, 4),
            "bbox_x1": round(lamp.representative_bbox[0], 1) if lamp.representative_bbox else "",
            "bbox_y1": round(lamp.representative_bbox[1], 1) if lamp.representative_bbox else "",
            "bbox_x2": round(lamp.representative_bbox[2], 1) if lamp.representative_bbox else "",
            "bbox_y2": round(lamp.representative_bbox[3], 1) if lamp.representative_bbox else "",
            "frame_count": lamp.frame_count,
            "first_frame": lamp.first_frame,
            "last_frame": lamp.last_frame,
            "frames_on": lamp.frames_on,
            "frames_off": lamp.frames_off,
            "on_fraction_pct": round(lamp.frames_on / observed_frames * 100.0, 2),
            "brightness_range": round(lamp.max_brightness - lamp.min_brightness, 2),
            "latitude": location.get("latitude", ""),
            "longitude": location.get("longitude", ""),
            "gps_accuracy_m": location.get("gps_accuracy_m", ""),
            "device_id": location.get("device_id", ""),
            "route_group": location.get("route_group", ""),
            "prior_candidate_lamp_id": (lamp.existence_prior or {}).get("candidate_lamp_id", ""),
            "prior_claim": prior_match.get("claim", ""),
            "prior_confidence": prior_match.get("confidence", ""),
            "prior_distance_m": prior_match.get("distance_m", ""),
            "prior_new_candidate": (lamp.existence_prior or {}).get("new_candidate", ""),
        })

    path = output_dir / "audit_report.csv"
    _write_csv(path, rows)
    return path


# ------------------------------------------------------------------ #
# Markdown report                                                     #
# ------------------------------------------------------------------ #

def write_markdown_report(
    output_dir: Path,
    lamps: list[AggregatedLamp],
    config: dict[str, Any],
    video_meta: dict[str, Any],
    filter_stats: dict[str, Any],
    eval_metrics: dict[str, Any] | None = None,
    location_prior_report: dict[str, Any] | None = None,
) -> Path:
    """Write a human-readable Markdown audit report."""
    lines: list[str] = []
    quality_summary = _lamp_quality_summary(lamps, filter_stats)
    detector_cfg = config.get("detector", {})

    # Title
    lines.append("# 🔦 Streetlight Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── Video metadata ───────────────────────────────────────────────
    lines.append("## 📹 Video Information")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Video Path | `{config.get('video_path', 'N/A')}` |")
    lines.append(f"| Resolution | {video_meta.get('width', '?')} × {video_meta.get('height', '?')} |")
    lines.append(f"| FPS | {video_meta.get('fps', '?')} |")
    lines.append(f"| Total Frames | {video_meta.get('frame_count', '?')} |")
    lines.append(f"| Model | `{detector_cfg.get('model_path', 'N/A')}` |")
    lines.append(f"| Target classes used | {_resolved_class_text(config)} |")
    lines.append("")

    # ── Summary counts ───────────────────────────────────────────────
    total = len(lamps)
    working = sum(1 for l in lamps if l.status == "working")
    off = sum(1 for l in lamps if l.status == "off")
    flickering = sum(1 for l in lamps if l.status == "flickering")

    lines.append("## 📊 Audit Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Total lamps detected** | **{total}** |")
    lines.append(f"| ✅ Lamps working (on) | {working} |")
    lines.append(f"| ❌ Lamps off / faulty | {off} |")
    lines.append(f"| ⚡ Lamps flickering | {flickering} |")
    if total > 0:
        lines.append(f"| Working rate | {quality_summary['working_rate_pct']:.1f}% |")
        lines.append(f"| Mean detector confidence | {quality_summary['mean_detector_confidence']:.3f} |")
        lines.append(f"| Median frames per final lamp | {quality_summary['median_track_frames']:.1f} |")
    lines.append("")

    # ── Brightness distribution ──────────────────────────────────────
    if lamps:
        all_avg = [l.avg_brightness for l in lamps]
        lines.append("## 💡 Brightness Distribution")
        lines.append("")
        lines.append(f"| Statistic | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Min avg brightness | {min(all_avg):.2f} |")
        lines.append(f"| Max avg brightness | {max(all_avg):.2f} |")
        lines.append(f"| Mean avg brightness | {sum(all_avg) / len(all_avg):.2f} |")
        lines.append(f"| Median avg brightness | {_median(all_avg):.2f} |")
        lines.append(f"| 25th percentile | {_percentile(all_avg, 0.25):.2f} |")
        lines.append(f"| 75th percentile | {_percentile(all_avg, 0.75):.2f} |")
        if len(all_avg) > 1:
            lines.append(f"| Std dev | {statistics.stdev(all_avg):.2f} |")
        lines.append(f"| Brightness threshold | {config.get('measurement', {}).get('brightness_threshold', '?')} |")
        lines.append("")

    # ── Filter effectiveness ─────────────────────────────────────────
    lines.append("## 🔍 Multi-Cue Filter Effectiveness")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total YOLO boxes returned | {filter_stats.get('total_model_detections', '?')} |")
    lines.append(f"| Target-class boxes before cues | {filter_stats.get('total_raw_detections', '?')} |")
    lines.append(f"| Non-target boxes suppressed | {filter_stats.get('non_target_detections_suppressed', 0)} |")
    lines.append(f"| Target boxes after cues | {filter_stats.get('total_kept_detections', '?')} |")
    rejection_rate = filter_stats.get("rejection_rate_pct", "?")
    lines.append(f"| Rejection rate | {rejection_rate}% |")
    lines.append(f"| Acceptance rate | {quality_summary['target_class_acceptance_rate_pct']:.2f}% |")
    temp = filter_stats.get("temporal_filtering", {})
    if temp:
        lines.append(f"| Tracks removed after temporal check | {temp.get('tracks_removed', '?')} |")
        lines.append(f"| Duplicate tracks merged | {temp.get('duplicates_merged', '?')} |")
    lines.append("")
    # Per-reason breakdown
    reasons = filter_stats.get("rejection_reasons", {})
    if reasons:
        lines.append("**Rejection breakdown:**")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|--------|-------|")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    if location_prior_report:
        query = location_prior_report.get("query", {})
        updates = location_prior_report.get("current_run_updates", {})
        summary = location_prior_report.get("prior_summary", {})
        lines.append("## Location Prior")
        lines.append("")
        lines.append("| Signal | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Query claim | {query.get('claim', 'not queried')} |")
        if query.get("match"):
            match = query["match"]
            lines.append(f"| Query candidate | {match.get('candidate_lamp_id', '')} |")
            lines.append(f"| Query confidence | {match.get('confidence', '')} |")
            lines.append(f"| Query distance | {match.get('distance_m', '')} m |")
        lines.append(f"| Known candidates in prior | {summary.get('known_lamp_candidates', 0)} |")
        lines.append(f"| Location evidence from this run | {updates.get('location_evidence_count', 0)} |")
        lines.append(f"| Matched existing candidates | {updates.get('matched_existing_candidates', 0)} |")
        lines.append(f"| New candidates created | {updates.get('new_candidates', 0)} |")
        if location_prior_report.get("updated_prior_path"):
            lines.append(f"| Updated prior | `{location_prior_report['updated_prior_path']}` |")
        lines.append("")

    lines.append("## 📈 Run Quality Signals")
    lines.append("")
    if not eval_metrics:
        lines.append(
            "**Accuracy metrics are not available for this run because no ground-truth labels were provided.** "
            "The signals below describe stability and confidence, but they are not a substitute for precision, recall, or mAP."
        )
        lines.append("")
    lines.append("| Signal | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Mean detector confidence | {quality_summary['mean_detector_confidence']:.3f} |")
    lines.append(f"| Median detector confidence | {quality_summary['median_detector_confidence']:.3f} |")
    lines.append(f"| Low-confidence final lamps (<0.5) | {quality_summary['low_confidence_lamps_lt_0_5']} |")
    lines.append(f"| Median track length | {quality_summary['median_track_frames']:.1f} frames |")
    lines.append(f"| Track length range | {quality_summary['min_track_frames']} - {quality_summary['max_track_frames']} frames |")
    lines.append(f"| Mean on-frame fraction | {quality_summary['mean_on_fraction_pct']:.1f}% |")
    lines.append(f"| Target detections per final lamp | {quality_summary['target_detections_per_final_lamp']:.2f} |")
    lines.append(f"| Non-target suppression rate | {quality_summary['non_target_suppression_rate_pct']:.2f}% |")
    lines.append("")

    # ── Per-lamp table ───────────────────────────────────────────────
    lines.append("## 📋 Per-Lamp Details")
    lines.append("")
    if lamps:
        lines.append("| # | Track ID | Status | Avg Brightness | On Frames | Confidence | Frames | Brightness Range | BBox (x1,y1,x2,y2) |")
        lines.append("|---|----------|--------|----------------|-----------|------------|--------|------------------|---------------------|")
        for i, lamp in enumerate(lamps, 1):
            status_icon = {"working": "✅", "off": "❌", "flickering": "⚡"}.get(lamp.status, "❓")
            bbox_str = ", ".join(f"{v:.0f}" for v in lamp.representative_bbox) if lamp.representative_bbox else "N/A"
            on_pct = lamp.frames_on / max(lamp.frames_on + lamp.frames_off, 1) * 100.0
            brightness_range = lamp.max_brightness - lamp.min_brightness
            lines.append(
                f"| {i} | {lamp.track_id} | {status_icon} {lamp.status} "
                f"| {lamp.avg_brightness:.1f} | {on_pct:.1f}% "
                f"| {lamp.confidence:.3f} | {lamp.frame_count} "
                f"| {brightness_range:.1f} "
                f"| ({bbox_str}) |"
            )
        lines.append("")
    else:
        lines.append("*No lamps detected.*")
        lines.append("")

    # ── Evaluation metrics ───────────────────────────────────────────
    lines.append("## 📏 Evaluation Metrics")
    lines.append("")
    if not eval_metrics:
        lines.append(
            "No ground-truth label directory was provided, so this run cannot report true accuracy, precision, recall, F1, or mAP. "
            "Use `--gt-labels /path/to/yolo_labels` to enable box-level evaluation."
        )
        lines.append("")
    else:
        context = eval_metrics.get("evaluation_context", {})
        if context:
            lines.append("### Evaluation Coverage")
            lines.append("")
            lines.append("| Item | Value |")
            lines.append("|------|-------|")
            lines.append(f"| Frames evaluated | {context.get('frames_evaluated', '?')} |")
            lines.append(f"| Frames with GT boxes | {context.get('frames_with_ground_truth', '?')} |")
            lines.append(f"| GT box instances | {context.get('ground_truth_instances', '?')} |")
            lines.append(f"| Raw target predictions | {context.get('raw_target_prediction_instances', '?')} |")
            lines.append(f"| Filtered target predictions | {context.get('filtered_target_prediction_instances', '?')} |")
            lines.append(f"| Unique predicted lamp tracks | {context.get('unique_lamps_after_tracking', '?')} |")
            lines.append("")

        det = eval_metrics.get("detection", {})
        frame_det = eval_metrics.get("frame_detection", {})
        if det:
            lines.append("### Detection Performance")
            lines.append("")
            lines.append("| Metric | Value | How to read it |")
            lines.append("|--------|-------|----------------|")
            lines.append(f"| **AP@0.5** | **{det.get('AP@0.5', '?')}** | Box match quality at IoU 0.5. Higher is better. |")
            lines.append(f"| **mAP@0.5:0.95** | **{det.get('mAP@0.5:0.95', '?')}** | Stricter localization score averaged over IoU thresholds. |")
            if frame_det:
                lines.append(f"| Precision @0.5 | {frame_det.get('precision', '?')} | Of predicted boxes, fraction that matched GT. Low means false positives. |")
                lines.append(f"| Recall @0.5 | {frame_det.get('recall', '?')} | Of GT boxes, fraction found. Low means missed lamps. |")
                lines.append(f"| F1 @0.5 | {frame_det.get('f1', '?')} | Balance between precision and recall. |")
                lines.append(f"| TP / FP / FN @0.5 | {frame_det.get('tp', '?')} / {frame_det.get('fp', '?')} / {frame_det.get('fn', '?')} | Matched, extra, and missed frame-level boxes. |")
            lines.append("")

            per_threshold = det.get("per_threshold_AP", {})
            if per_threshold:
                lines.append("### AP by IoU Threshold")
                lines.append("")
                lines.append("| IoU | AP |")
                lines.append("|-----|----|")
                for threshold, ap_value in per_threshold.items():
                    lines.append(f"| {threshold} | {ap_value} |")
                lines.append("")

        before_after = eval_metrics.get("before_after_filtering", {})
        if before_after:
            lines.append("### Filter Impact on Detection Metrics")
            lines.append("")
            lines.append("| Metric | Before Cues | After Cues | Delta |")
            lines.append("|--------|-------------|------------|-------|")
            for metric_name in ["precision", "recall", "f1"]:
                before_val = before_after.get("before", {}).get(metric_name, 0.0)
                after_val = before_after.get("after", {}).get(metric_name, 0.0)
                delta = after_val - before_val if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)) else "?"
                delta_text = f"{delta:+.4f}" if isinstance(delta, float) else delta
                lines.append(f"| {metric_name.title()} | {before_val} | {after_val} | {delta_text} |")
            lines.append("")

        frame_counts = eval_metrics.get("frame_instance_counts")
        if frame_counts:
            lines.append("### Frame-Instance Count Check")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| GT box instances | {frame_counts.get('ground_truth_instances', '?')} |")
            lines.append(f"| Filtered prediction instances | {frame_counts.get('filtered_prediction_instances', '?')} |")
            lines.append(f"| Count error | {frame_counts.get('count_error_pct', '?')}% |")
            lines.append(f"| Extra prediction instances (FP @0.5) | {frame_counts.get('extra_instances_fp_at_iou_0_5', '?')} |")
            lines.append(f"| Missed GT instances (FN @0.5) | {frame_counts.get('missed_instances_fn_at_iou_0_5', '?')} |")
            lines.append("")

        status = eval_metrics.get("status_classification")
        if status:
            lines.append("### Brightness / Status Classification")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Accuracy | {status.get('accuracy', '?')} |")
            lines.append(f"| Precision (working) | {status.get('precision', '?')} |")
            lines.append(f"| Recall (working) | {status.get('recall', '?')} |")
            lines.append(f"| F1 | {status.get('f1', '?')} |")
            lines.append("")
            cm = status.get("confusion_matrix", {})
            if cm:
                lines.append("| Actual \\ Predicted | Working | Off/Faulty |")
                lines.append("|--------------------|---------|------------|")
                lines.append(f"| Working | {cm.get('tp', '?')} | {cm.get('fn', '?')} |")
                lines.append(f"| Off/Faulty | {cm.get('fp', '?')} | {cm.get('tn', '?')} |")
                lines.append("")

    # ── Uncertain cases ──────────────────────────────────────────────
    uncertain = [l for l in lamps if l.confidence < 0.5 or l.brightness_std > 40]
    if uncertain:
        lines.append("## ⚠️ Flagged Uncertain Cases")
        lines.append("")
        lines.append("The following lamps have low confidence or high brightness variance:")
        lines.append("")
        lines.append("| Track ID | Status | Confidence | Brightness Std |")
        lines.append("|----------|--------|------------|----------------|")
        for lamp in uncertain:
            lines.append(
                f"| {lamp.track_id} | {lamp.status} "
                f"| {lamp.confidence:.3f} | {lamp.brightness_std:.2f} |"
            )
        lines.append("")

    # ── Configuration ────────────────────────────────────────────────
    lines.append("## ⚙️ Pipeline Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2, default=str))
    lines.append("```")
    lines.append("")

    # ── Footer ───────────────────────────────────────────────────────
    lines.append("---")
    lines.append("*Report generated by the Streetlight Audit Pipeline*")
    lines.append("")

    path = output_dir / "audit_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
