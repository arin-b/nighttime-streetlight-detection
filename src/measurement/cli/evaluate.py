from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_measurement.evaluation.metrics import summarize_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or summarize measurement-block reports.")
    parser.add_argument("--pred", required=True, help="reports.json produced by measure-clip or measure-batch.")
    parser.add_argument("--gt", default=None, help="Optional ground-truth path reserved for future metric expansion.")
    parser.add_argument("--out", required=True, help="Output evaluation directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = json.loads(Path(args.pred).read_text(encoding="utf-8-sig"))
    summary = summarize_reports(reports)
    if args.gt:
        summary["ground_truth"] = str(Path(args.gt))
        summary["note"] = "Ground-truth metric expansion is scaffolded; current evaluator summarizes predictions."
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
