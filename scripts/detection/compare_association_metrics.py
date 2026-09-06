"""Compare centre displacement and IoU on manually verified lamp identities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

from tracker.association_metrics import consecutive_negative_pairs, consecutive_positive_pairs


METRICS = ("centre_distance", "normalized_centre_distance", "one_minus_iou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="CSV with frame,lamp_id,x1,y1,x2,y2.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-negative-pairs", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * value / 100.0
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def statistics(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, float]]:
    collected = list(rows)
    return {
        metric: {
            "count": len(values := [float(row[metric]) for row in collected]),
            "mean": mean(values) if values else 0.0,
            "median": median(values) if values else 0.0,
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "p75": percentile(values, 75),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
        }
        for metric in METRICS
    }


def write_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row["pair_type"] == "positive"]
        figure, axis = plt.subplots()
        axis.hist(values, bins="auto")
        axis.set(title=f"Positive-pair {metric}", xlabel=metric, ylabel="count")
        figure.savefig(output_dir / f"{metric}_histogram.png", bbox_inches="tight")
        plt.close(figure)
    if any(row["pair_type"] == "negative" for row in rows):
        for metric in ("centre_distance", "one_minus_iou"):
            figure, axis = plt.subplots()
            for kind in ("positive", "negative"):
                axis.hist([float(row[metric]) for row in rows if row["pair_type"] == kind], bins="auto", alpha=0.55, label=kind)
            axis.set(title=f"{metric}: positive vs negative", xlabel=metric, ylabel="count")
            axis.legend()
            figure.savefig(output_dir / f"{metric}_positive_negative.png", bbox_inches="tight")
            plt.close(figure)


def main() -> None:
    args = parse_args()
    with Path(args.input_csv).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame", "lamp_id", "x1", "y1", "x2", "y2"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Input CSV must contain: {', '.join(sorted(required))}")
    results = consecutive_positive_pairs(rows)
    if args.include_negative_pairs:
        results.extend(consecutive_negative_pairs(rows))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ("pair_type", "lamp_id", "frame_t", "frame_t1", "centre_distance", "normalized_centre_distance", "iou", "one_minus_iou")
    with (output_dir / "association_metric_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    summary = {"consecutive_same_lamp_pairs": sum(row["pair_type"] == "positive" for row in results), "positive": statistics(row for row in results if row["pair_type"] == "positive")}
    if args.include_negative_pairs:
        summary["negative"] = statistics(row for row in results if row["pair_type"] == "negative")
    print(json.dumps(summary, indent=2))
    if not args.no_plots:
        write_plots(results, output_dir)


if __name__ == "__main__":
    main()
