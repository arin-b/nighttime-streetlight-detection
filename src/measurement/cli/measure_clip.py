from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_measurement.pipeline import run_clip_to_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run measurement-block inference for one clip manifest.")
    parser.add_argument("--input", required=True, help="Clip manifest JSON.")
    parser.add_argument("--out", required=True, help="Output run directory.")
    parser.add_argument("--measurement-run-id", default=None, help="Optional stable measurement run ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = run_clip_to_directory(args.input, args.out, args.measurement_run_id)
    print(json.dumps({"out": str(Path(args.out)), "reports": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
