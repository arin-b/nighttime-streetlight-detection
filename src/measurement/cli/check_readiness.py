from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_measurement.training.readiness import check_dataset_readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a measurement dataset is ready for stronger-model training.")
    parser.add_argument("--dataset", required=True, help="Prepared measurement dataset root.")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    parser.add_argument("--skip-models", action="store_true", help="Do not materialize/check pretrained model assets.")
    parser.add_argument("--no-annotations", action="store_true", help="Only validate clip manifests and tracks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_dataset_readiness(
        args.dataset,
        require_annotations=not args.no_annotations,
        ensure_models=not args.skip_models,
    ).to_dict()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
