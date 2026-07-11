from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reliability-gate AUC and threshold behavior by slice.")
    parser.add_argument("--input", required=True, help="CSV with at least dataset_id, reliability_score, and is_correct.")
    parser.add_argument("--output", required=True, help="Markdown output path for the evaluation report.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Acceptance threshold for operating-point metrics.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def roc_auc(rows: list[dict]) -> float:
    labeled = []
    for row in rows:
        if row.get("is_correct", "") == "":
            continue
        labeled.append((float(row["reliability_score"]), int(float(row["is_correct"]))))
    positives = [item for item in labeled if item[1] == 1]
    negatives = [item for item in labeled if item[1] == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def operating_point(rows: list[dict], threshold: float) -> dict[str, float]:
    labeled = []
    for row in rows:
        if row.get("is_correct", "") == "":
            continue
        labeled.append((float(row["reliability_score"]), int(float(row["is_correct"]))))
    tp = fp = tn = fn = 0
    for score, label in labeled:
        predicted_positive = score >= threshold
        if predicted_positive and label == 1:
            tp += 1
        elif predicted_positive and label == 0:
            fp += 1
        elif not predicted_positive and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def slice_rows(rows: list[dict]) -> dict[str, list[dict]]:
    slices: dict[str, list[dict]] = {"pooled": rows}
    for row in rows:
        dataset_id = row.get("dataset_id", "").strip() or "unknown"
        slices.setdefault(dataset_id, []).append(row)
    return slices


def format_metric(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    slices = slice_rows(rows)

    lines = [
        "# Reliability Gate Evaluation",
        "",
        f"- Input: `{args.input}`",
        f"- Acceptance threshold: `{args.threshold:.2f}`",
        "",
        "| Slice | Labeled Rows | AUC | Precision | Recall | Specificity | TP | FP | TN | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for slice_name, slice_data in sorted(slices.items()):
        labeled_count = sum(1 for row in slice_data if row.get("is_correct", "") != "")
        auc = roc_auc(slice_data)
        op = operating_point(slice_data, args.threshold)
        lines.append(
            "| "
            + " | ".join(
                [
                    slice_name,
                    str(labeled_count),
                    format_metric(auc),
                    format_metric(op["precision"]),
                    format_metric(op["recall"]),
                    format_metric(op["specificity"]),
                    str(op["tp"]),
                    str(op["fp"]),
                    str(op["tn"]),
                    str(op["fn"]),
                ]
            )
            + " |"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
