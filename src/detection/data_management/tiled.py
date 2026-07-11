from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiled YOLO dataset for small-object streetlight training.")
    parser.add_argument("--dataset-root", required=True, help="Source YOLO dataset root with images/labels train|valid|test.")
    parser.add_argument("--output-root", required=True, help="Destination tiled dataset root.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile width and height in pixels.")
    parser.add_argument("--stride", type=int, default=768, help="Tile stride in pixels.")
    parser.add_argument("--min-box-retained-frac", type=float, default=0.6, help="Minimum fraction of original box area retained inside a tile.")
    parser.add_argument("--min-box-side-px", type=float, default=6.0, help="Minimum retained box width and height in pixels.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_dataset_yaml(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def load_labels(path: Path, image_width: int, image_height: int) -> list[list[float]]:
    if not path.exists():
        return []
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, x_c, y_c, w, h = line.split()
        x_c = float(x_c) * image_width
        y_c = float(y_c) * image_height
        w = float(w) * image_width
        h = float(h) * image_height
        boxes.append([x_c - (w / 2.0), y_c - (h / 2.0), w, h])
    return boxes


def yolo_line(box: list[float], tile_size: int) -> str:
    x, y, w, h = box
    x_c = (x + (w / 2.0)) / tile_size
    y_c = (y + (h / 2.0)) / tile_size
    return f"0 {x_c:.6f} {y_c:.6f} {w / tile_size:.6f} {h / tile_size:.6f}"


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def intersect_box(box: list[float], tile_x: int, tile_y: int, tile_size: int) -> tuple[list[float], float] | None:
    x, y, w, h = box
    x1 = max(x, tile_x)
    y1 = max(y, tile_y)
    x2 = min(x + w, tile_x + tile_size)
    y2 = min(y + h, tile_y + tile_size)
    if x2 <= x1 or y2 <= y1:
        return None
    retained = (x2 - x1) * (y2 - y1)
    total = w * h
    if total <= 0:
        return None
    return [x1 - tile_x, y1 - tile_y, x2 - x1, y2 - y1], retained / total


def copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required for tile generation. Install it in the training environment.") from exc

    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root / "images" / "train")
    ensure_dir(output_root / "labels" / "train")

    for split in ("valid", "test"):
        copy_dir(dataset_root / "images" / split, output_root / "images" / split)
        copy_dir(dataset_root / "labels" / split, output_root / "labels" / split)

    manifest_rows: list[dict[str, str]] = []
    train_images = sorted((dataset_root / "images" / "train").glob("*"))
    tile_count = 0
    kept_boxes = 0
    positive_tiles = 0
    negative_tiles = 0

    for image_path in train_images:
        label_path = dataset_root / "labels" / "train" / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            boxes = load_labels(label_path, width, height)
            xs = tile_starts(width, args.tile_size, args.stride)
            ys = tile_starts(height, args.tile_size, args.stride)
            for tile_x in xs:
                for tile_y in ys:
                    tile_boxes: list[list[float]] = []
                    for box in boxes:
                        clipped = intersect_box(box, tile_x, tile_y, args.tile_size)
                        if clipped is None:
                            continue
                        new_box, retained_frac = clipped
                        if retained_frac < args.min_box_retained_frac:
                            continue
                        if new_box[2] < args.min_box_side_px or new_box[3] < args.min_box_side_px:
                            continue
                        tile_boxes.append(new_box)

                    tile = image.crop((tile_x, tile_y, tile_x + args.tile_size, tile_y + args.tile_size))
                    tile_name = f"{image_path.stem}__x{tile_x:04d}_y{tile_y:04d}.jpg"
                    tile_image_path = output_root / "images" / "train" / tile_name
                    tile_label_path = output_root / "labels" / "train" / f"{Path(tile_name).stem}.txt"
                    tile.save(tile_image_path, format="JPEG", quality=95)
                    tile_label_path.write_text("\n".join(yolo_line(box, args.tile_size) for box in tile_boxes), encoding="utf-8")

                    tile_count += 1
                    kept_boxes += len(tile_boxes)
                    if tile_boxes:
                        positive_tiles += 1
                    else:
                        negative_tiles += 1
                    manifest_rows.append(
                        {
                            "source_image": str(image_path),
                            "tile_image": str(tile_image_path),
                            "tile_label": str(tile_label_path),
                            "tile_x": str(tile_x),
                            "tile_y": str(tile_y),
                            "retained_boxes": str(len(tile_boxes)),
                        }
                    )

    dataset_yaml = "\n".join(
        [
            f"path: {output_root.as_posix()}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            "",
            "names:",
            "  0: streetlight",
            "",
        ]
    )
    (output_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    ensure_dir(output_root / "manifests")
    with (output_root / "manifests" / "tile_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_image", "tile_image", "tile_label", "tile_x", "tile_y", "retained_boxes"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_root / "manifests" / "tile_summary.json").write_text(
        json.dumps(
            {
                "tile_size": args.tile_size,
                "stride": args.stride,
                "tile_count": tile_count,
                "positive_tiles": positive_tiles,
                "negative_tiles": negative_tiles,
                "retained_boxes": kept_boxes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "tile_count": tile_count, "positive_tiles": positive_tiles}, indent=2))


if __name__ == "__main__":
    main()
