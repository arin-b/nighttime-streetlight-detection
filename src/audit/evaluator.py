"""Three-level evaluation metrics (PDF §7).

1. Detection metrics  — Precision, Recall, F1, AP@0.5, mAP@0.5:0.95
2. Status classification — Accuracy, Precision, Recall, F1, confusion matrix
3. Audit-level — lamp count error, working-lamp count error
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ------------------------------------------------------------------ #
# Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class DetectionMetrics:
    """Standard object-detection scores at a single IoU threshold."""
    iou_threshold: float
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class StatusClassificationMetrics:
    """Binary classification metrics for working vs off."""
    tp: int = 0  # correctly identified working
    fp: int = 0  # predicted working, actually off
    tn: int = 0  # correctly identified off
    fn: int = 0  # predicted off, actually working

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion_matrix": {
                "tp": self.tp,
                "fp": self.fp,
                "tn": self.tn,
                "fn": self.fn,
            },
        }


@dataclass
class AuditMetrics:
    """Audit-level count metrics."""
    n_true: int = 0
    n_pred: int = 0
    n_on_true: int = 0
    n_on_pred: int = 0

    @property
    def count_error_pct(self) -> float:
        if self.n_true == 0:
            return 0.0
        return abs(self.n_pred - self.n_true) / self.n_true * 100

    @property
    def working_count_error_pct(self) -> float:
        if self.n_on_true == 0:
            return 0.0
        return abs(self.n_on_pred - self.n_on_true) / self.n_on_true * 100

    @property
    def missed_lamps(self) -> int:
        return max(0, self.n_true - self.n_pred)

    @property
    def extra_lamps(self) -> int:
        return max(0, self.n_pred - self.n_true)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_true": self.n_true,
            "total_predicted": self.n_pred,
            "lamp_count_error_pct": round(self.count_error_pct, 2),
            "working_true": self.n_on_true,
            "working_predicted": self.n_on_pred,
            "working_count_error_pct": round(self.working_count_error_pct, 2),
            "missed_lamps_fn": self.missed_lamps,
            "extra_lamps_fp": self.extra_lamps,
        }


# ------------------------------------------------------------------ #
# IoU computation                                                     #
# ------------------------------------------------------------------ #

def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------ #
# Per-frame detection evaluation                                      #
# ------------------------------------------------------------------ #

def evaluate_frame_detections(
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    gt_boxes: list[list[float]],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Match predicted boxes to ground truth at a given IoU threshold.

    Uses greedy matching: sort predictions by score descending, match each
    to the highest-IoU unmatched GT box.
    """
    metrics = DetectionMetrics(iou_threshold=iou_threshold)

    if not gt_boxes and not pred_boxes:
        return metrics
    if not gt_boxes:
        metrics.fp = len(pred_boxes)
        return metrics
    if not pred_boxes:
        metrics.fn = len(gt_boxes)
        return metrics

    # Sort predictions by score descending
    order = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
    matched_gt = set()

    for idx in order:
        best_iou = 0.0
        best_gt = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou_val = _iou(pred_boxes[idx], gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = gi
        if best_iou >= iou_threshold and best_gt >= 0:
            metrics.tp += 1
            matched_gt.add(best_gt)
        else:
            metrics.fp += 1

    metrics.fn = len(gt_boxes) - len(matched_gt)
    return metrics


# ------------------------------------------------------------------ #
# AP computation (area under precision-recall curve)                  #
# ------------------------------------------------------------------ #

def compute_ap(
    all_pred_boxes: list[list[list[float]]],
    all_pred_scores: list[list[float]],
    all_gt_boxes: list[list[list[float]]],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision at a single IoU threshold across all frames.

    Uses the PASCAL VOC 11-point interpolation method.
    """
    # Collect all predictions with frame index
    preds = []
    for frame_idx, (boxes, scores) in enumerate(zip(all_pred_boxes, all_pred_scores)):
        for box, score in zip(boxes, scores):
            preds.append((score, frame_idx, box))

    # Sort by score descending
    preds.sort(key=lambda x: x[0], reverse=True)

    # Track which GT boxes have been matched per frame
    gt_matched: dict[int, set[int]] = {i: set() for i in range(len(all_gt_boxes))}
    total_gt = sum(len(gt) for gt in all_gt_boxes)

    if total_gt == 0:
        return 0.0

    tp_list = []
    fp_list = []

    for score, frame_idx, pred_box in preds:
        gt_boxes = all_gt_boxes[frame_idx]
        best_iou = 0.0
        best_gt = -1
        for gi, gt_box in enumerate(gt_boxes):
            if gi in gt_matched[frame_idx]:
                continue
            iou_val = _iou(pred_box, gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = gi
        if best_iou >= iou_threshold and best_gt >= 0:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[frame_idx].add(best_gt)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Cumulative sums
    tp_cum = np.cumsum(tp_list).astype(float)
    fp_cum = np.cumsum(fp_list).astype(float)

    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / total_gt

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += float(np.max(prec_at_recall))
    ap /= 11.0

    return ap


def compute_map(
    all_pred_boxes: list[list[list[float]]],
    all_pred_scores: list[list[float]],
    all_gt_boxes: list[list[list[float]]],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Compute mAP@0.5 and mAP@0.5:0.95."""
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    ap_50 = compute_ap(all_pred_boxes, all_pred_scores, all_gt_boxes, 0.5)

    aps = [
        compute_ap(all_pred_boxes, all_pred_scores, all_gt_boxes, t)
        for t in iou_thresholds
    ]
    map_50_95 = sum(aps) / len(aps) if aps else 0.0

    return {
        "AP@0.5": round(ap_50, 4),
        "mAP@0.5:0.95": round(map_50_95, 4),
        "per_threshold_AP": {f"{t:.2f}": round(ap, 4) for t, ap in zip(iou_thresholds, aps)},
    }


# ------------------------------------------------------------------ #
# Ground-truth loader                                                 #
# ------------------------------------------------------------------ #

def load_gt_boxes_yolo(
    labels_dir: str | Path,
    frame_index: int,
    vid_stride: int,
    frame_width: int,
    frame_height: int,
) -> list[list[float]]:
    """Load YOLO-format ground-truth boxes for a given frame.

    Expected label filename: ``frame_NNNNNN.txt``
    Each line: ``class_id x_center y_center width height`` (normalised).
    Returns xyxy boxes in pixel coordinates.
    """
    labels_dir = Path(labels_dir)
    actual_frame = ((frame_index - 1) * max(1, vid_stride)) + 1
    label_path = labels_dir / f"frame_{actual_frame:06d}.txt"

    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # Convert normalised xywh → pixel xyxy
        x1 = (xc - w / 2) * frame_width
        y1 = (yc - h / 2) * frame_height
        x2 = (xc + w / 2) * frame_width
        y2 = (yc + h / 2) * frame_height
        boxes.append([x1, y1, x2, y2])

    return boxes
