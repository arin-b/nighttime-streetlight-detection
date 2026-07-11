from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_measurement.training.readiness import check_dataset_readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a joint-training run manifest for the stronger research model.")
    parser.add_argument("--dataset", required=True, help="Prepared measurement dataset root.")
    parser.add_argument("--config", required=True, help="Joint training config path.")
    parser.add_argument("--out", required=True, help="Output training run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the joint-training plan manifest.")
    parser.add_argument("--skip-readiness", action="store_true", help="Do not validate dataset/model readiness before writing the plan.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    readiness = None
    if not args.skip_readiness:
        readiness = check_dataset_readiness(args.dataset, require_annotations=True, ensure_models=True)
    manifest = {
        "run_type": "joint_training",
        "dataset": str(Path(args.dataset)),
        "config": str(Path(args.config)),
        "status": "ready" if readiness is None or readiness.ready else "blocked",
        "dry_run": bool(args.dry_run),
        "readiness": readiness.to_dict() if readiness else None,
        "note": "Joint fine-tuning requires trained module checkpoints and a PyTorch trainer implementation.",
    }
    (out / "joint_training_plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if readiness is not None and not readiness.ready:
        print(json.dumps(manifest, indent=2))
        raise SystemExit("Training readiness failed. See joint_training_plan.json for issues.")
    if not args.dry_run:
        checkpoint = {
            "checkpoint_type": "initialized_joint_checkpoint",
            "dataset": str(Path(args.dataset)),
            "config": str(Path(args.config)),
            "pretrained_assets": readiness.model_assets if readiness else {},
            "status": "initialized_not_optimized",
            "next_step": "attach joint PyTorch fine-tuning loop after module checkpoints exist",
        }
        (out / "joint_checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        manifest["status"] = "initialized"
        manifest["checkpoint"] = "joint_checkpoint.json"
        (out / "joint_training_plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
