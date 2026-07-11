from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from rbccps_od.training.dataset_yaml import resolve_dataset_yaml_for_runtime

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a one-class YOLOv26 streetlight detector.")
    parser.add_argument("--model", required=True, help="Path to pretrained YOLOv26 weights.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml generated from the cleaned seed corpus.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs", help="Ultralytics project output directory.")
    parser.add_argument("--name", default="streetlight_detector_v1", help="Run name.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience in epochs.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic augmentation in late epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration without starting training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    resolved = {
        "model": str(Path(args.model).resolve()),
        "data": str(data_path),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "cache": args.cache,
        "close_mosaic": args.close_mosaic,
    }
    if args.dry_run:
        for key, value in resolved.items():
            print(f"{key}={value}")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install the remote requirements package before starting training."
        ) from exc

    data_arg = resolved["data"]
    if data_path.suffix.lower() in {".yaml", ".yml"} and data_path.exists():
        data_arg = str(resolve_dataset_yaml_for_runtime(data_path))

    model = YOLO(resolved["model"])
    model.train(
        data=data_arg,
        imgsz=resolved["imgsz"],
        epochs=resolved["epochs"],
        batch=resolved["batch"],
        device=resolved["device"],
        project=resolved["project"],
        name=resolved["name"],
        patience=resolved["patience"],
        workers=resolved["workers"],
        cache=resolved["cache"],
        close_mosaic=resolved["close_mosaic"],
    )


if __name__ == "__main__":
    main()
