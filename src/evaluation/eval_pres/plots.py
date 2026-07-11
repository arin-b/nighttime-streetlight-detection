from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import EvaluationDetails, ILLUMINATION_ORDER, STATUS_LABELS
from .schemas import MetricResult


def write_plots(metrics: list[MetricResult], details: EvaluationDetails, out: str | Path) -> dict[str, str]:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    artifacts["metric_scorecard"] = str(_metric_scorecard(metrics, out_path / "metric_scorecard.png"))
    if details.precision_recall_curve:
        artifacts["detection_pr_curve"] = str(_pr_curve(details, out_path / "detection_pr_curve.png"))
    if details.status_pairs:
        artifacts["status_confusion_matrix"] = str(
            _confusion_matrix(details.status_pairs, STATUS_LABELS, out_path / "status_confusion_matrix.png", "Lamp Status Confusion Matrix")
        )
    if details.illumination_pairs:
        artifacts["illumination_confusion_matrix"] = str(
            _confusion_matrix(details.illumination_pairs, ILLUMINATION_ORDER, out_path / "illumination_confusion_matrix.png", "Illumination Class Confusion Matrix")
        )
    if details.per_lamp_rows:
        artifacts["per_lamp_metric_relationships"] = str(_metric_relationships(details, out_path / "per_lamp_metric_relationships.png"))
    if details.track_to_gt_ids:
        artifacts["tracking_timeline"] = str(_tracking_timeline(details, out_path / "tracking_timeline.png"))
    return artifacts


def _metric_scorecard(metrics: list[MetricResult], path: Path) -> Path:
    computed = [metric for metric in metrics if metric.status == "computed" and isinstance(metric.value, (int, float))]
    if not computed:
        _empty_plot(path, "No computed metrics")
        return path
    labels = [metric.name for metric in computed]
    values = [_display_value(metric) for metric in computed]
    colors = ["#2ca25f" if metric.direction == "maximize" else "#de2d26" for metric in computed]
    height = max(4, 0.35 * len(computed))
    fig, ax = plt.subplots(figsize=(11, height))
    y = np.arange(len(computed))
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Metric value; lower is better for red bars, higher is better for green bars")
    ax.set_title("RBCCPS Evaluation Metric Scorecard")
    for idx, metric in enumerate(computed):
        ax.text(values[idx], idx, f" {metric.value} {metric.unit}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _display_value(metric: MetricResult) -> float:
    value = float(metric.value or 0.0)
    if metric.unit in {"fraction", "AP", "F1", "IoU", "score"}:
        return value
    return value


def _pr_curve(details: EvaluationDetails, path: Path) -> Path:
    recall = [row[2] for row in details.precision_recall_curve]
    precision = [row[1] for row in details.precision_recall_curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(recall, precision, where="post", color="#3182bd", linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Detector Precision-Recall Curve at IoU 0.50")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _confusion_matrix(pairs: tuple[tuple[str, str], ...], labels: tuple[str, ...], path: Path, title: str) -> Path:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for true, pred in pairs:
        if true in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[true], label_to_idx[pred]] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="#111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _metric_relationships(details: EvaluationDetails, path: Path) -> Path:
    rows = details.per_lamp_rows
    x = [row["detector_score"] for row in rows]
    y = [row["measurement_score"] for row in rows]
    c = [row["confidence"] for row in rows]
    s = [45 + 60 * row["matched"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(x, y, c=c, s=s, cmap="viridis", alpha=0.8, edgecolor="#333", linewidth=0.3)
    ax.set_xlabel("Mean Detector Score")
    ax.set_ylabel("Measurement Useful-Illumination Score")
    ax.set_title("Detector Score vs Measurement Score")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Report confidence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _tracking_timeline(details: EvaluationDetails, path: Path) -> Path:
    track_ids = sorted(details.track_to_gt_ids)
    lengths = [len(details.track_to_gt_ids[track]) for track in track_ids]
    unique_gt = [len(set(details.track_to_gt_ids[track])) for track in track_ids]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.22 * len(track_ids))))
    y = np.arange(len(track_ids))
    ax.barh(y, lengths, color="#74a9cf", label="matched frames")
    ax.scatter(unique_gt, y, color="#cb181d", label="unique GT identities")
    ax.set_yticks(y, labels=track_ids, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Tracking Match Summary")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, title, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

