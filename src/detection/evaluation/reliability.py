from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FEATURE_NAMES = [
    "detector_confidence",
    "augmentation_agreement",
    "temporal_persistence",
    "geometry_score",
    "stability_score",
]

DEFAULT_RULE_WEIGHTS = {
    "detector_confidence": 0.35,
    "augmentation_agreement": 0.20,
    "temporal_persistence": 0.20,
    "geometry_score": 0.15,
    "stability_score": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit or score the hybrid annotation reliability gate.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a lightweight logistic gate from reviewed calibration data.")
    fit_parser.add_argument("--input", required=True, help="CSV with feature columns and an is_correct label column.")
    fit_parser.add_argument("--output-model", required=True, help="JSON output path for the fitted model.")
    fit_parser.add_argument("--iterations", type=int, default=1500, help="Training iterations for logistic regression.")
    fit_parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for logistic regression.")

    score_parser = subparsers.add_parser("score", help="Score candidate annotations with the hybrid gate.")
    score_parser.add_argument("--input", required=True, help="CSV with candidate features.")
    score_parser.add_argument("--output", required=True, help="CSV output path with reliability scores and bands.")
    score_parser.add_argument("--model", help="Optional fitted model JSON from the fit command.")
    score_parser.add_argument("--accept-threshold", type=float, default=0.95, help="Auto-accept threshold.")
    score_parser.add_argument("--review-threshold", type=float, default=0.60, help="Manual-review threshold.")
    return parser.parse_args()


def sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rule_score(row: dict) -> float:
    score = 0.0
    for name, weight in DEFAULT_RULE_WEIGHTS.items():
        score += float(row.get(name, 0.0) or 0.0) * weight
    return max(0.0, min(1.0, score))


def fit_logistic(rows: list[dict], iterations: int, learning_rate: float) -> dict:
    weights = {name: 0.0 for name in FEATURE_NAMES}
    bias = 0.0

    training_rows = []
    for row in rows:
        if row.get("is_correct", "") == "":
            continue
        training_rows.append((row, float(row["is_correct"])))
    if not training_rows:
        raise SystemExit("No labeled rows were found in the fit input.")

    for _ in range(iterations):
        grad_w = {name: 0.0 for name in FEATURE_NAMES}
        grad_b = 0.0
        for row, label in training_rows:
            linear = bias
            for name in FEATURE_NAMES:
                linear += weights[name] * float(row.get(name, 0.0) or 0.0)
            pred = sigmoid(linear)
            error = pred - label
            for name in FEATURE_NAMES:
                grad_w[name] += error * float(row.get(name, 0.0) or 0.0)
            grad_b += error

        sample_count = float(len(training_rows))
        for name in FEATURE_NAMES:
            weights[name] -= learning_rate * (grad_w[name] / sample_count)
        bias -= learning_rate * (grad_b / sample_count)

    return {"feature_names": FEATURE_NAMES, "weights": weights, "bias": bias}


def model_score(row: dict, model: dict) -> float:
    linear = float(model["bias"])
    for name in model["feature_names"]:
        linear += float(model["weights"][name]) * float(row.get(name, 0.0) or 0.0)
    return sigmoid(linear)


def fit_command(args: argparse.Namespace) -> None:
    rows = load_rows(Path(args.input))
    model = fit_logistic(rows, iterations=args.iterations, learning_rate=args.lr)
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def score_command(args: argparse.Namespace) -> None:
    rows = load_rows(Path(args.input))
    model = None
    if args.model:
        model = json.loads(Path(args.model).read_text(encoding="utf-8"))

    output_rows = []
    for row in rows:
        rule_based = rule_score(row)
        learned = model_score(row, model) if model else rule_based
        hybrid = (rule_based + learned) / 2.0
        if hybrid >= args.accept_threshold:
            band = "auto-accept"
        elif hybrid >= args.review_threshold:
            band = "manual-review"
        else:
            band = "auto-reject"
        merged_row = dict(row)
        merged_row["rule_score"] = f"{rule_based:.6f}"
        merged_row["learned_score"] = f"{learned:.6f}"
        merged_row["reliability_score"] = f"{hybrid:.6f}"
        merged_row["acceptance_band"] = band
        output_rows.append(merged_row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        fit_command(args)
    else:
        score_command(args)


if __name__ == "__main__":
    main()
