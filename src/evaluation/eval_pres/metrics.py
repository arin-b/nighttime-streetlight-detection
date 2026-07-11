from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schemas import BoxRecord, EvaluationInputs, MetricResult, ReportRecord


IOU_MATCH_THRESHOLD = 0.50
STATUS_LABELS = ("on", "dim", "off", "saturated", "occluded", "flicker")
ILLUMINATION_ORDER = ("unknown", "poor", "marginal", "adequate")


@dataclass(frozen=True)
class DetectionMatch:
    pred_index: int
    gt_index: int | None
    iou: float
    threshold: float


@dataclass(frozen=True)
class EvaluationDetails:
    matches_050: tuple[DetectionMatch, ...]
    track_to_gt_ids: dict[str, list[str]]
    report_pairs: tuple[tuple[ReportRecord, BoxRecord], ...]
    precision_recall_curve: tuple[tuple[float, float, float], ...]
    status_pairs: tuple[tuple[str, str], ...]
    illumination_pairs: tuple[tuple[str, str], ...]
    per_lamp_rows: tuple[dict[str, float], ...]


def evaluate(inputs: EvaluationInputs) -> tuple[list[MetricResult], EvaluationDetails]:
    matches_050 = tuple(match_predictions(inputs.predictions, inputs.ground_truth, IOU_MATCH_THRESHOLD))
    track_to_gt_ids = _track_to_gt_ids(inputs.predictions, inputs.ground_truth, matches_050)
    report_pairs = tuple(_report_gt_pairs(inputs.reports, inputs.ground_truth, track_to_gt_ids))
    status_pairs = tuple(_status_pairs(report_pairs))
    illumination_pairs = tuple(_illumination_pairs(report_pairs))
    pr_curve = tuple(_precision_recall_curve(inputs.predictions, inputs.ground_truth, IOU_MATCH_THRESHOLD))
    per_lamp_rows = tuple(_per_lamp_rows(inputs.predictions, inputs.reports, matches_050))

    metrics = [
        _application_detection_rate(inputs.ground_truth, matches_050),
        _duplicate_physical_lamp_rate(inputs.ground_truth, track_to_gt_ids),
        _track_identity_switch_count(track_to_gt_ids),
        _track_fragmentation_rate(inputs.ground_truth, track_to_gt_ids),
        _inventory_match_accuracy(report_pairs),
        _false_positives_per_km(inputs.predictions, matches_050, inputs.route_distance_km),
        _latency(inputs.latency_seconds),
        _model_size(inputs.model_size_mb),
        _tracked_lamp_detection_recall(inputs.ground_truth, matches_050),
        _duplicate_physical_lamp_track_rate(inputs.ground_truth, track_to_gt_ids),
        _track_identity_switch_count(track_to_gt_ids, section="Detection and Tracking"),
        _detector_ap(inputs.predictions, inputs.ground_truth, (0.50,), "Detector AP50"),
        _detector_ap(inputs.predictions, inputs.ground_truth, tuple(np.arange(0.50, 0.751, 0.05)), "Detector AP50-75"),
        _detector_ap(inputs.predictions, inputs.ground_truth, tuple(np.arange(0.50, 0.951, 0.05)), "Detector AP50-95"),
        _detector_precision(inputs.predictions, inputs.ground_truth, matches_050),
        _detector_recall(inputs.ground_truth, matches_050),
        _detector_size_error(inputs.predictions, inputs.ground_truth, matches_050),
        _affected_region_iou(report_pairs),
        _lamp_status_macro_f1(status_pairs),
        _false_attribution_under_confounders(report_pairs),
        _poor_as_adequate_rate(illumination_pairs),
        _spatial_coverage_bias(report_pairs),
        _temporal_stability(inputs.reports),
    ]
    details = EvaluationDetails(
        matches_050=matches_050,
        track_to_gt_ids=track_to_gt_ids,
        report_pairs=report_pairs,
        precision_recall_curve=pr_curve,
        status_pairs=status_pairs,
        illumination_pairs=illumination_pairs,
        per_lamp_rows=per_lamp_rows,
    )
    return metrics, details


def match_predictions(predictions: tuple[BoxRecord, ...], ground_truth: tuple[BoxRecord, ...], threshold: float) -> list[DetectionMatch]:
    if not predictions or not ground_truth:
        return []
    gt_by_frame: dict[int, list[int]] = defaultdict(list)
    for index, gt in enumerate(ground_truth):
        gt_by_frame[gt.frame_id].append(index)
    order = sorted(range(len(predictions)), key=lambda idx: predictions[idx].score, reverse=True)
    used_gt: set[int] = set()
    matches: list[DetectionMatch] = []
    for pred_index in order:
        pred = predictions[pred_index]
        best_gt = None
        best_iou = 0.0
        for gt_index in gt_by_frame.get(pred.frame_id, []):
            if gt_index in used_gt:
                continue
            overlap = bbox_iou(pred.bbox_xyxy, ground_truth[gt_index].bbox_xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_gt = gt_index
        if best_gt is not None and best_iou >= threshold:
            used_gt.add(best_gt)
            matches.append(DetectionMatch(pred_index, best_gt, best_iou, threshold))
        else:
            matches.append(DetectionMatch(pred_index, None, best_iou, threshold))
    return matches


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def polygon_bbox_iou(a: tuple[tuple[float, float], ...], b: tuple[tuple[float, float], ...]) -> float:
    return bbox_iou(_polygon_bounds(a), _polygon_bounds(b))


def _polygon_bounds(poly: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _average_precision(predictions: tuple[BoxRecord, ...], ground_truth: tuple[BoxRecord, ...], threshold: float) -> float | None:
    if not ground_truth:
        return None
    matches = match_predictions(predictions, ground_truth, threshold)
    if not matches:
        return 0.0
    tp = []
    fp = []
    for match in matches:
        tp.append(1.0 if match.gt_index is not None else 0.0)
        fp.append(1.0 if match.gt_index is None else 0.0)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, len(ground_truth))
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return float(_integrate_pr(recalls, precisions))


def _integrate_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1]))


def _precision_recall_curve(
    predictions: tuple[BoxRecord, ...],
    ground_truth: tuple[BoxRecord, ...],
    threshold: float,
) -> list[tuple[float, float, float]]:
    if not ground_truth:
        return []
    matches = match_predictions(predictions, ground_truth, threshold)
    curve = []
    tp = fp = 0
    for match in matches:
        if match.gt_index is None:
            fp += 1
        else:
            tp += 1
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, len(ground_truth))
        score = predictions[match.pred_index].score
        curve.append((float(score), float(precision), float(recall)))
    return curve


def _computed(name: str, section: str, direction: str, value: float | int, unit: str, description: str) -> MetricResult:
    if isinstance(value, float):
        value = round(value, 6)
    return MetricResult(name, section, direction, value, unit, description)


def _na(name: str, section: str, direction: str, unit: str, description: str, reason: str) -> MetricResult:
    return MetricResult(name, section, direction, None, unit, description, status="not_available", reason=reason)


def _gt_physical_ids(gt: Iterable[BoxRecord]) -> set[str]:
    return {item.physical_lamp_id or f"frame:{item.frame_id}:box:{item.bbox_xyxy}" for item in gt}


def _matched_gt_ids(ground_truth: tuple[BoxRecord, ...], matches: tuple[DetectionMatch, ...]) -> set[str]:
    ids = set()
    for match in matches:
        if match.gt_index is not None:
            gt = ground_truth[match.gt_index]
            ids.add(gt.physical_lamp_id or f"frame:{gt.frame_id}:box:{gt.bbox_xyxy}")
    return ids


def _track_to_gt_ids(
    predictions: tuple[BoxRecord, ...],
    ground_truth: tuple[BoxRecord, ...],
    matches: tuple[DetectionMatch, ...],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for match in sorted(matches, key=lambda item: (predictions[item.pred_index].track_id or "", predictions[item.pred_index].frame_id)):
        pred = predictions[match.pred_index]
        if pred.track_id is None or match.gt_index is None:
            continue
        gt = ground_truth[match.gt_index]
        result[pred.track_id].append(gt.physical_lamp_id or f"frame:{gt.frame_id}:box:{gt.bbox_xyxy}")
    return dict(result)


def _application_detection_rate(gt: tuple[BoxRecord, ...], matches: tuple[DetectionMatch, ...]) -> MetricResult:
    ids = _gt_physical_ids(gt)
    if not ids:
        return _na("Application Detection Rate", "Overall", "maximize", "fraction", "Real lamps that survive into measurement input.", "requires ground-truth physical lamps")
    return _computed("Application Detection Rate", "Overall", "maximize", len(_matched_gt_ids(gt, matches)) / len(ids), "fraction", "Real lamps that survive detection, tracking, and manifest creation.")


def _duplicate_physical_lamp_rate(gt: tuple[BoxRecord, ...], track_to_gt_ids: dict[str, list[str]], section: str = "Overall") -> MetricResult:
    ids = _gt_physical_ids(gt)
    if not ids:
        return _na("Duplicate Physical Lamp Rate", section, "minimize", "fraction", "Real lamps split into multiple predicted tracks or reports.", "requires ground-truth physical lamps")
    gt_to_tracks: dict[str, set[str]] = defaultdict(set)
    for track_id, gt_ids in track_to_gt_ids.items():
        for gt_id in gt_ids:
            gt_to_tracks[gt_id].add(track_id)
    duplicate_lamps = sum(1 for gt_id in ids if len(gt_to_tracks.get(gt_id, set())) > 1)
    return _computed("Duplicate Physical Lamp Rate", section, "minimize", duplicate_lamps / len(ids), "fraction", "How often one real lamp becomes multiple tracks or reports.")


def _track_identity_switch_count(track_to_gt_ids: dict[str, list[str]], section: str = "Overall") -> MetricResult:
    switches = 0
    any_track = False
    for gt_ids in track_to_gt_ids.values():
        if gt_ids:
            any_track = True
        switches += sum(1 for a, b in zip(gt_ids, gt_ids[1:]) if a != b)
    if not any_track:
        return _na("Track Identity Switch Count", section, "minimize", "count", "Track identity changes across frames.", "requires matched tracks and ground-truth identities")
    return _computed("Track Identity Switch Count", section, "minimize", switches, "count", "Number of times a lamp track changes identity across frames.")


def _track_fragmentation_rate(gt: tuple[BoxRecord, ...], track_to_gt_ids: dict[str, list[str]]) -> MetricResult:
    base = _duplicate_physical_lamp_rate(gt, track_to_gt_ids, section="Overall")
    return MetricResult("Track Fragmentation Rate", "Overall", "minimize", base.value, "fraction", "Real lamp observations split into multiple track fragments.", base.status, base.reason)


def _inventory_match_accuracy(report_pairs: tuple[tuple[ReportRecord, BoxRecord], ...]) -> MetricResult:
    eligible = [(report, gt) for report, gt in report_pairs if gt.inventory_id and report.mapped_lamp_id]
    if not eligible:
        return _na("Inventory Match Accuracy", "Overall", "maximize", "fraction", "Reports matched to inventory physical lamp IDs.", "requires report mapped_lamp_id and ground-truth inventory_id")
    correct = sum(1 for report, gt in eligible if report.mapped_lamp_id == gt.inventory_id)
    return _computed("Inventory Match Accuracy", "Overall", "maximize", correct / len(eligible), "fraction", "Fraction of reported lamps correctly matched to inventory physical lamp IDs.")


def _false_positives_per_km(
    predictions: tuple[BoxRecord, ...],
    matches: tuple[DetectionMatch, ...],
    route_distance_km: float | None,
) -> MetricResult:
    if route_distance_km is None or route_distance_km <= 0:
        return _na("False Positives Per Kilometer", "Overall", "minimize", "count/km", "False lamp reports per kilometer.", "requires route distance in km")
    unmatched_track_ids = {
        predictions[match.pred_index].track_id or f"pred:{match.pred_index}"
        for match in matches
        if match.gt_index is None
    }
    return _computed("False Positives Per Kilometer", "Overall", "minimize", len(unmatched_track_ids) / route_distance_km, "count/km", "Number of false lamp reports per kilometer of route.")


def _latency(value: float | None) -> MetricResult:
    if value is None:
        return _na("End-to-End Latency", "Overall", "minimize", "seconds", "Time from input to usable report output.", "requires --latency-seconds")
    return _computed("End-to-End Latency", "Overall", "minimize", value, "seconds", "Time from frame or clip input to usable report output.")


def _model_size(value: float | None) -> MetricResult:
    if value is None:
        return _na("Model Size", "Overall", "minimize", "MB", "Storage footprint of detector and measurement assets.", "requires --model-path")
    return _computed("Model Size", "Overall", "minimize", value, "MB", "Storage footprint of detector and measurement models/assets.")


def _tracked_lamp_detection_recall(gt: tuple[BoxRecord, ...], matches: tuple[DetectionMatch, ...]) -> MetricResult:
    base = _application_detection_rate(gt, matches)
    return MetricResult("Tracked Lamp Detection Recall", "Detection and Tracking", "maximize", base.value, "fraction", "Real lamps detected and tracked at least once.", base.status, base.reason)


def _duplicate_physical_lamp_track_rate(gt: tuple[BoxRecord, ...], track_to_gt_ids: dict[str, list[str]]) -> MetricResult:
    base = _duplicate_physical_lamp_rate(gt, track_to_gt_ids, section="Detection and Tracking")
    return MetricResult("Duplicate Physical Lamp Track Rate", "Detection and Tracking", "minimize", base.value, "fraction", "How often one real lamp becomes multiple duplicate tracks.", base.status, base.reason)


def _detector_ap(predictions: tuple[BoxRecord, ...], gt: tuple[BoxRecord, ...], thresholds: tuple[float, ...], name: str) -> MetricResult:
    if not gt:
        return _na(name, "Detection and Tracking", "maximize", "AP", "Detector average precision.", "requires ground-truth boxes")
    values = [_average_precision(predictions, gt, float(threshold)) for threshold in thresholds]
    return _computed(name, "Detection and Tracking", "maximize", float(np.mean([v for v in values if v is not None])), "AP", f"Detection average precision over IoU thresholds {thresholds[0]:.2f}-{thresholds[-1]:.2f}.")


def _detector_precision(predictions: tuple[BoxRecord, ...], gt: tuple[BoxRecord, ...], matches: tuple[DetectionMatch, ...]) -> MetricResult:
    if not gt:
        return _na("Detector Precision", "Detection and Tracking", "maximize", "fraction", "Predicted lamp detections that are true lamps.", "requires ground-truth boxes")
    if not predictions:
        return _computed("Detector Precision", "Detection and Tracking", "maximize", 0.0, "fraction", "Predicted lamp detections that are true lamps.")
    tp = sum(1 for match in matches if match.gt_index is not None)
    return _computed("Detector Precision", "Detection and Tracking", "maximize", tp / len(predictions), "fraction", "Fraction of predicted lamp detections that are true lamps.")


def _detector_recall(gt: tuple[BoxRecord, ...], matches: tuple[DetectionMatch, ...]) -> MetricResult:
    if not gt:
        return _na("Detector Recall", "Detection and Tracking", "maximize", "fraction", "Real lamps that are detected.", "requires ground-truth boxes")
    tp = sum(1 for match in matches if match.gt_index is not None)
    return _computed("Detector Recall", "Detection and Tracking", "maximize", tp / len(gt), "fraction", "Fraction of real lamp boxes that are detected.")


def _detector_size_error(predictions: tuple[BoxRecord, ...], gt: tuple[BoxRecord, ...], matches: tuple[DetectionMatch, ...]) -> MetricResult:
    errors = []
    for match in matches:
        if match.gt_index is None:
            continue
        pred_box = predictions[match.pred_index].bbox_xyxy
        gt_box = gt[match.gt_index].bbox_xyxy
        pw, ph = _box_wh(pred_box)
        gw, gh = _box_wh(gt_box)
        errors.append((abs(pw - gw) / max(gw, 1e-6) + abs(ph - gh) / max(gh, 1e-6)) / 2)
    if not errors:
        return _na("Detector Size Error", "Detection and Tracking", "minimize", "relative_error", "Predicted-vs-true box size error.", "requires matched detections and ground-truth boxes")
    return _computed("Detector Size Error", "Detection and Tracking", "minimize", float(np.mean(errors)), "relative_error", "Mean relative difference between predicted and true box width/height.")


def _box_wh(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def _report_gt_pairs(
    reports: tuple[ReportRecord, ...],
    gt: tuple[BoxRecord, ...],
    track_to_gt_ids: dict[str, list[str]],
) -> list[tuple[ReportRecord, BoxRecord]]:
    by_id = {item.physical_lamp_id: item for item in gt if item.physical_lamp_id}
    pairs = []
    for report in reports:
        ids = track_to_gt_ids.get(report.lamp_track_id, [])
        gt_id = _mode(ids)
        if gt_id and gt_id in by_id:
            pairs.append((report, by_id[gt_id]))
    return pairs


def _mode(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(counts, key=counts.get)


def _affected_region_iou(report_pairs: tuple[tuple[ReportRecord, BoxRecord], ...]) -> MetricResult:
    values = []
    for report, gt in report_pairs:
        if report.affected_region_polygon and gt.affected_region_polygon:
            values.append(polygon_bbox_iou(report.affected_region_polygon, gt.affected_region_polygon))
    if not values:
        return _na("Affected Public-Space Region IoU", "Measurement", "maximize", "IoU", "Predicted served public-space region overlap.", "requires predicted and ground-truth affected-region polygons")
    return _computed("Affected Public-Space Region IoU", "Measurement", "maximize", float(np.mean(values)), "IoU", "Mean IoU of predicted and true affected public-space regions.")


def _status_pairs(report_pairs: tuple[tuple[ReportRecord, BoxRecord], ...]) -> list[tuple[str, str]]:
    return [
        (str(gt.status), str(report.status_label))
        for report, gt in report_pairs
        if gt.status in STATUS_LABELS and report.status_label in STATUS_LABELS
    ]


def _lamp_status_macro_f1(pairs: tuple[tuple[str, str], ...]) -> MetricResult:
    if not pairs:
        return _na("Lamp Emission Status Macro F1", "Measurement", "maximize", "F1", "Status classification quality.", "requires ground-truth and predicted status labels")
    scores = []
    for label in STATUS_LABELS:
        tp = sum(1 for true, pred in pairs if true == label and pred == label)
        fp = sum(1 for true, pred in pairs if true != label and pred == label)
        fn = sum(1 for true, pred in pairs if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return _computed("Lamp Emission Status Macro F1", "Measurement", "maximize", float(np.mean(scores)), "F1", "Macro F1 across on, dim, off, saturated, occluded, and flicker.")


def _false_attribution_under_confounders(report_pairs: tuple[tuple[ReportRecord, BoxRecord], ...]) -> MetricResult:
    eligible = []
    for report, gt in report_pairs:
        if gt.confounder_present is not True:
            continue
        predicted_target_credit = _predicts_target_credit(report)
        if predicted_target_credit is not None:
            eligible.append((report, gt, predicted_target_credit))
    if not eligible:
        return _na("False Target-Lamp Attribution Rate Under Confounders", "Measurement", "minimize", "fraction", "Non-target light incorrectly credited to target lamp.", "requires confounder labels and predicted attribution evidence")
    false_count = sum(1 for _, gt, predicted in eligible if predicted and gt.target_attribution_correct is False)
    return _computed("False Target-Lamp Attribution Rate Under Confounders", "Measurement", "minimize", false_count / len(eligible), "fraction", "Headlights, shopfronts, signs, reflections, or other lamps credited to target streetlight.")


def _predicts_target_credit(report: ReportRecord) -> bool | None:
    if report.target_attribution_score is not None:
        return report.target_attribution_score >= 0.5
    if report.attribution_class:
        return report.attribution_class in {"certain", "likely_target", "target_lamp_primary", "mixed"}
    return None


def _illumination_pairs(report_pairs: tuple[tuple[ReportRecord, BoxRecord], ...]) -> list[tuple[str, str]]:
    values = []
    for report, gt in report_pairs:
        pred = _normalize_illumination(report.overall_category)
        true = _normalize_illumination(gt.illumination_class)
        if pred and true:
            values.append((true, pred))
    return values


def _normalize_illumination(label: str | None) -> str | None:
    if not label:
        return None
    value = str(label).lower().strip()
    if value in {"dark", "underlit", "bad", "poor"}:
        return "poor"
    if value in {"limited", "marginal"}:
        return "marginal"
    if value in {"good", "excellent", "acceptable", "adequate"}:
        return "adequate"
    if value == "unknown":
        return "unknown"
    return None


def _poor_as_adequate_rate(pairs: tuple[tuple[str, str], ...]) -> MetricResult:
    poor = [(true, pred) for true, pred in pairs if true == "poor"]
    if not poor:
        return _na("Poor-As-Adequate Illumination Error Rate", "Measurement", "minimize", "fraction", "Bad lighting reported as acceptable.", "requires poor/underlit ground-truth illumination labels")
    false_adequate = sum(1 for _, pred in poor if pred == "adequate")
    return _computed("Poor-As-Adequate Illumination Error Rate", "Measurement", "minimize", false_adequate / len(poor), "fraction", "Probability of bad lighting being reported as acceptable.")


def _spatial_coverage_bias(report_pairs: tuple[tuple[ReportRecord, BoxRecord], ...]) -> MetricResult:
    diffs = [
        abs(float(report.affected_region_area_fraction) - float(gt.served_area_fraction))
        for report, gt in report_pairs
        if report.affected_region_area_fraction is not None and gt.served_area_fraction is not None
    ]
    if not diffs:
        return _na("Spatial Coverage Bias", "Measurement", "minimize", "absolute_fraction_error", "Predicted served-area size bias.", "requires predicted and ground-truth served-area fractions")
    return _computed("Spatial Coverage Bias", "Measurement", "minimize", float(np.mean(diffs)), "absolute_fraction_error", "Mean absolute difference between predicted and true served-area fraction.")


def _temporal_stability(reports: tuple[ReportRecord, ...]) -> MetricResult:
    groups: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        key = report.mapped_lamp_id or report.lamp_track_id
        if key and report.overall_score is not None:
            groups[key].append(float(report.overall_score))
    stds = [float(np.std(values)) for values in groups.values() if len(values) >= 2]
    if not stds:
        return _na("Temporal Report Stability For Same Lamp", "Measurement", "maximize", "score", "Repeated-report consistency for same lamp.", "requires repeated reports for the same lamp")
    stability = max(0.0, min(1.0, 1.0 - float(np.mean(stds))))
    return _computed("Temporal Report Stability For Same Lamp", "Measurement", "maximize", stability, "score", "Consistency of repeated measurements for the same physical lamp.")


def _per_lamp_rows(
    predictions: tuple[BoxRecord, ...],
    reports: tuple[ReportRecord, ...],
    matches: tuple[DetectionMatch, ...],
) -> list[dict[str, float]]:
    matched_tracks = {predictions[match.pred_index].track_id for match in matches if match.gt_index is not None}
    by_track = {report.lamp_track_id: report for report in reports}
    rows = []
    for track_id, report in by_track.items():
        if not track_id:
            continue
        detections = [pred for pred in predictions if pred.track_id == track_id]
        if not detections:
            continue
        rows.append(
            {
                "detector_score": float(np.mean([det.score for det in detections])),
                "matched": 1.0 if track_id in matched_tracks else 0.0,
                "measurement_score": float(report.overall_score or 0.0),
                "confidence": float(report.confidence or 0.0),
            }
        )
    return rows

