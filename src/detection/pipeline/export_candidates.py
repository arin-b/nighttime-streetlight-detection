from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rbccps_od.training.ultralytics_adapter import UltralyticsAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YOLOv26 detector and export candidate annotations.")
    parser.add_argument("--model", required=True, help="Path to fine-tuned detector weights.")
    parser.add_argument("--images", required=True, help="Input image directory or manifest CSV with image_path column.")
    parser.add_argument("--output", required=True, help="Output CSV for candidate annotations.")
    parser.add_argument("--conf", type=float, default=0.05, help="Detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    return parser.parse_args()


def iter_image_paths(image_arg: str) -> list[Path]:
    path = Path(image_arg)
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*.jpg"))
    rows: list[Path] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = row.get("image_path")
            if image_path:
                rows.append(Path(image_path))
    return rows


def main() -> None:
    args = parse_args()
    image_paths = iter_image_paths(args.images)
    adapter = UltralyticsAdapter()
    model = adapter.load(args.model)

    fieldnames = [
        "image_path",
        "box_index",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "detector_confidence",
        "class_id",
        "class_name",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_path in image_paths:
            results = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box_index, box in enumerate(result.boxes):
                    xyxy = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    writer.writerow(
                        {
                            "image_path": str(image_path),
                            "box_index": box_index,
                            "bbox_x1": xyxy[0],
                            "bbox_y1": xyxy[1],
                            "bbox_x2": xyxy[2],
                            "bbox_y2": xyxy[3],
                            "detector_confidence": float(box.conf[0].item()),
                            "class_id": class_id,
                            "class_name": class_name,
                        }
                    )


if __name__ == "__main__":
    main()
