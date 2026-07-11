from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .io import build_inputs
from .metrics import evaluate
from .plots import write_plots
from .run_metadata import write_run_sidecars
from .schemas import MetricResult


def evaluate_to_directory(
    manifest: str | Path,
    reports: str | Path,
    out: str | Path,
    ground_truth: str | Path | None = None,
    route_distance_km: float | None = None,
    latency_seconds: float | None = None,
    model_paths: list[str | Path] | None = None,
    run_name: str = "raw_pretrained_coco_untrained_eval",
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(
        manifest=manifest,
        reports=reports,
        ground_truth=ground_truth,
        route_distance_km=route_distance_km,
        latency_seconds=latency_seconds,
        model_paths=model_paths or [],
    )
    metrics, details = evaluate(inputs)
    plots = write_plots(metrics, details, out / "plots")
    summary = {
        "run_name": run_name,
        "model_declaration": "evaluation_declared_on_raw_pretrained_coco_weights_not_fine_tuned_or_domain_adapted",
        "inputs": {
            "manifest": str(Path(manifest)),
            "reports": str(Path(reports)),
            "ground_truth": str(Path(ground_truth)) if ground_truth else None,
            "route_distance_km": inputs.route_distance_km,
            "latency_seconds": inputs.latency_seconds,
            "model_size_mb": inputs.model_size_mb,
        },
        "counts": {
            "predicted_boxes": len(inputs.predictions),
            "ground_truth_boxes": len(inputs.ground_truth),
            "measurement_reports": len(inputs.reports),
            "matched_detections_iou_050": sum(1 for match in details.matches_050 if match.gt_index is not None),
        },
        "metrics": [metric.to_dict() for metric in metrics],
        "plots": plots,
        "notes": [
            "Metrics requiring labels are marked not_available instead of estimated from predictions.",
            "Affected-region IoU uses polygon bounding-box IoU unless dense masks are provided in a future evaluator.",
            "This evaluator is intended for raw pretrained/untrained pipeline presentation runs and later domain-adapted comparisons.",
        ],
    }
    sidecars = write_run_sidecars(
        out,
        run_type="evaluate",
        parameters={
            "manifest": str(Path(manifest)),
            "reports": str(Path(reports)),
            "ground_truth": str(Path(ground_truth)) if ground_truth else None,
            "route_distance_km": route_distance_km,
            "latency_seconds": latency_seconds,
            "model_paths": [str(Path(path)) for path in (model_paths or [])],
            "run_name": run_name,
        },
        summary={
            "computed_metrics": sum(metric.status == "computed" for metric in metrics),
            "metric_count": len(metrics),
            "plots": plots,
        },
    )
    summary["run_sidecars"] = sidecars
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_metrics_csv(out / "metrics.csv", metrics)
    _write_metric_status_csv(out / "metric_status.csv", metrics)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RBCCPS detection, tracking, measurement, and auditing metrics.")
    parser.add_argument("--manifest", required=True, help="Measurement clip_manifest.json with detector/tracker outputs.")
    parser.add_argument("--reports", required=True, help="Measurement reports.json.")
    parser.add_argument("--ground-truth", help="Optional ground-truth JSON with boxes, IDs, status, and measurement labels.")
    parser.add_argument("--route-distance-km", type=float, help="Route distance in kilometers for FP/km.")
    parser.add_argument("--latency-seconds", type=float, help="End-to-end processing latency in seconds.")
    parser.add_argument("--model-path", action="append", default=[], help="Model file or directory. Can be repeated.")
    parser.add_argument("--out", required=True, help="Output evaluation directory.")
    parser.add_argument("--run-name", default="raw_pretrained_coco_untrained_eval", help="Name shown in summary metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_to_directory(
        manifest=args.manifest,
        reports=args.reports,
        out=args.out,
        ground_truth=args.ground_truth,
        route_distance_km=args.route_distance_km,
        latency_seconds=args.latency_seconds,
        model_paths=args.model_path,
        run_name=args.run_name,
    )
    print(json.dumps({"out": str(Path(args.out)), "computed_metrics": sum(m["status"] == "computed" for m in summary["metrics"]), "plots": summary["plots"]}, indent=2))


def _write_metrics_csv(path: Path, metrics: list[MetricResult]) -> None:
    fieldnames = ["section", "metric", "direction", "value", "unit", "description"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            row = metric.to_dict()
            writer.writerow({key: row[key] for key in fieldnames})


def _write_metric_status_csv(path: Path, metrics: list[MetricResult]) -> None:
    fieldnames = ["section", "metric", "status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            row = metric.to_dict()
            writer.writerow({key: row[key] for key in fieldnames})


if __name__ == "__main__":
    main()
