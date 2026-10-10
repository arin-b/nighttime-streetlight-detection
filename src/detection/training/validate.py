from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rbccps_od.training.dataset_yaml import resolve_dataset_yaml_for_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a one-class YOLOv26 streetlight detector.")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv26 weights.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output-json", help="Optional path for machine-readable validation metrics.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _metric_value(result: Any, names: tuple[str, ...]) -> float | None:
    for name in names:
        current = result
        found = True
        for part in name.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                found = False
                break
        if found:
            try:
                return float(current)
            except (TypeError, ValueError):
                continue
    return None


def validation_summary(result: Any, payload: dict[str, Any]) -> dict[str, Any]:
    box = getattr(result, "box", None)
    speed = getattr(result, "speed", None)
    return {
        "config": payload,
        "metrics": {
            "precision": _metric_value(box, ("mp", "p")),
            "recall": _metric_value(box, ("mr", "r")),
            "map50": _metric_value(box, ("map50",)),
            "map50_95": _metric_value(box, ("map", "map5095")),
            "f1": _f1(_metric_value(box, ("mp", "p")), _metric_value(box, ("mr", "r"))),
        },
        "speed": speed if isinstance(speed, dict) else {},
    }


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 6)


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
    from rbccps_od.training.ultralytics_adapter import UltralyticsAdapter

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
    summary = validation_summary(result, payload)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
