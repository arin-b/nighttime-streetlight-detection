from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_od.training.dataset_yaml import resolve_dataset_yaml_for_runtime
from rbccps_od.training.ultralytics_adapter import UltralyticsAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a one-class YOLOv26 streetlight detector.")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv26 weights.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "model": str(Path(args.model).resolve()),
        "data": str(resolve_dataset_yaml_for_runtime(Path(args.data))),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    adapter = UltralyticsAdapter()
    result = adapter.validate(
        payload["model"],
        data=payload["data"],
        imgsz=payload["imgsz"],
        batch=payload["batch"],
        device=payload["device"],
        split=payload["split"],
        conf=payload["conf"],
        iou=payload["iou"],
    )
    print(result)


if __name__ == "__main__":
    main()
