from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_od.models.yolo26 import YOLO26Detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current YOLOv26 baseline detector.")
    parser.add_argument("--model", help="Optional explicit model checkpoint path.")
    parser.add_argument("--image", required=True, help="Image path for inference.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "image": str(Path(args.image).resolve()),
        "model": str(Path(args.model).resolve()) if args.model else None,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    detector = YOLO26Detector(checkpoint_path=args.model) if args.model else YOLO26Detector()
    result = detector.predict(payload["image"], conf=args.conf, iou=args.iou, device=args.device)
    print(result)


if __name__ == "__main__":
    main()
