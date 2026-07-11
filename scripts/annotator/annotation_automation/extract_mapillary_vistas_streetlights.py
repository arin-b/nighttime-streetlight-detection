from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


CANONICAL_STREETLIGHT_LABELS = {
    "street-light",
    "street light",
    "object--street-light",
    "street_light",
    "streetlight",
}


@dataclass
class SplitLayout:
    source_name: str
    source_images_dir: Path
    source_polygons_dir: Path
    yolo_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Mapillary Vistas street-light annotations into YOLO format.")
    parser.add_argument("--dataset-root", required=True, help="Root directory of an extracted Mapillary Vistas dataset copy.")
    parser.add_argument("--output-root", required=True, help="Destination directory for the extracted YOLO dataset.")
    parser.add_argument("--include-negatives", action="store_true", help="Include images with zero street-light instances as empty-label negatives.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into the YOLO dataset. If omitted, image copies are skipped and only labels/manifests are written.")
    parser.add_argument("--min-box-area", type=float, default=0.0, help="Discard boxes smaller than this pixel area.")
    return parser.parse_args()


def resolve_split_layouts(dataset_root: Path) -> list[SplitLayout]:
    candidates = [
        ("training", "train"),
        ("validation", "valid"),
        ("testing", "test"),
    ]
    layouts: list[SplitLayout] = []
    for source_name, yolo_name in candidates:
        images_candidates = [
            dataset_root / source_name / "images",
            dataset_root / source_name / "imgs",
        ]
        polygons_candidates = [
            dataset_root / source_name / "v2.0" / "polygons",
            dataset_root / source_name / "polygons",
        ]
        images_dir = next((path for path in images_candidates if path.exists()), None)
        polygons_dir = next((path for path in polygons_candidates if path.exists()), None)
        if images_dir and polygons_dir:
            layouts.append(SplitLayout(source_name, images_dir, polygons_dir, yolo_name))
    if not layouts:
        raise FileNotFoundError(
            "Could not find Mapillary split folders. Expected paths like "
            "'training/images' and 'training/v2.0/polygons' under --dataset-root."
        )
    return layouts


def canonicalize_label(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.strip().lower().replace("_", "-")


def polygon_points(annotation: dict) -> list[tuple[float, float]]:
    polygon = annotation.get("polygon")
    if isinstance(polygon, list) and polygon:
        if isinstance(polygon[0], dict):
            if "x" in polygon[0] and "y" in polygon[0]:
                return [(float(point["x"]), float(point["y"])) for point in polygon]
        if isinstance(polygon[0], (list, tuple)) and len(polygon[0]) >= 2:
            return [(float(point[0]), float(point[1])) for point in polygon]

    poly2d = annotation.get("poly2d")
    if isinstance(poly2d, list) and poly2d:
        first = poly2d[0]
        vertices = first.get("vertices") if isinstance(first, dict) else None
        if isinstance(vertices, list) and vertices:
            return [(float(point[0]), float(point[1])) for point in vertices]

    return []


def bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return x0, y0, width, height


def image_dimensions(annotation_blob: dict) -> tuple[int, int]:
    width = annotation_blob.get("width") or annotation_blob.get("imgWidth") or annotation_blob.get("imageWidth")
    height = annotation_blob.get("height") or annotation_blob.get("imgHeight") or annotation_blob.get("imageHeight")
    if width is None or height is None:
        raise ValueError("Annotation JSON is missing image width/height metadata.")
    return int(width), int(height)


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def yolo_line(box: tuple[float, float, float, float], image_width: int, image_height: int) -> str:
    x, y, w, h = box
    cx = (x + (w / 2.0)) / image_width
    cy = (y + (h / 2.0)) / image_height
    nw = w / image_width
    nh = h / image_height
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    layouts = resolve_split_layouts(dataset_root)

    images_root = output_root / "images"
    labels_root = output_root / "labels"
    manifests_root = output_root / "manifests"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    split_stats: dict[str, dict[str, int]] = {}

    for layout in layouts:
        split_stats[layout.yolo_name] = {"images": 0, "positives": 0, "negatives": 0, "boxes": 0}
        (images_root / layout.yolo_name).mkdir(parents=True, exist_ok=True)
        (labels_root / layout.yolo_name).mkdir(parents=True, exist_ok=True)

        for polygon_file in sorted(layout.source_polygons_dir.glob("*.json")):
            annotation_blob = json.loads(polygon_file.read_text(encoding="utf-8"))
            image_width, image_height = image_dimensions(annotation_blob)
            image_path = find_image(layout.source_images_dir, polygon_file.stem)
            if image_path is None:
                continue

            boxes: list[tuple[float, float, float, float]] = []
            for annotation in annotation_blob.get("objects", []):
                raw_label = annotation.get("label") or annotation.get("category") or annotation.get("name")
                label = canonicalize_label(raw_label)
                if label not in CANONICAL_STREETLIGHT_LABELS:
                    continue
                bbox = bbox_from_points(polygon_points(annotation))
                if bbox is None:
                    continue
                if bbox[2] * bbox[3] < args.min_box_area:
                    continue
                boxes.append(bbox)

            if not boxes and not args.include_negatives:
                continue

            target_image = images_root / layout.yolo_name / image_path.name
            target_label = labels_root / layout.yolo_name / f"{image_path.stem}.txt"
            if args.copy_images:
                shutil.copy2(image_path, target_image)

            label_lines = [yolo_line(box, image_width, image_height) for box in boxes]
            target_label.write_text("\n".join(label_lines), encoding="utf-8")

            split_stats[layout.yolo_name]["images"] += 1
            split_stats[layout.yolo_name]["boxes"] += len(boxes)
            if boxes:
                split_stats[layout.yolo_name]["positives"] += 1
            else:
                split_stats[layout.yolo_name]["negatives"] += 1

            manifest_rows.append(
                {
                    "split": layout.yolo_name,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "label_path": str(target_label),
                    "streetlight_count": str(len(boxes)),
                    "width": str(image_width),
                    "height": str(image_height),
                    "source_polygon_json": str(polygon_file),
                }
            )

    with (manifests_root / "mapillary_streetlight_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "image_name",
                "image_path",
                "label_path",
                "streetlight_count",
                "width",
                "height",
                "source_polygon_json",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    dataset_yaml_lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/valid",
    ]
    if (images_root / "test").exists():
        dataset_yaml_lines.append("test: images/test")
    dataset_yaml_lines.extend(["", "names:", "  0: streetlight", ""])
    (output_root / "dataset.yaml").write_text("\n".join(dataset_yaml_lines), encoding="utf-8")

    summary = {
        "source_dataset": "Mapillary Vistas",
        "streetlight_labels": sorted(CANONICAL_STREETLIGHT_LABELS),
        "include_negatives": bool(args.include_negatives),
        "copy_images": bool(args.copy_images),
        "min_box_area": args.min_box_area,
        "splits": split_stats,
    }
    (manifests_root / "mapillary_streetlight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
