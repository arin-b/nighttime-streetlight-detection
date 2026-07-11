from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image


STREETLIGHT_LABEL_ID = "/m/033rq4"
STREETLIGHT_LABEL_NAME = "Street light"

BOX_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "valid": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}

IMAGE_INDEX_URLS = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "valid": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}

NEGATIVE_LABEL_URLS = {
    "train": "https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv",
    "valid": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv",
}


@dataclass
class SampleRecord:
    split: str
    image_id: str
    image_url: str
    original_url: str
    rotation: int
    boxes: list[tuple[float, float, float, float]]
    is_negative: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract an Open Images Street light subset into YOLO format.")
    parser.add_argument("--output-root", required=True, help="Destination directory for the extracted YOLO dataset.")
    parser.add_argument("--cache-root", required=True, help="Directory for downloaded metadata caches.")
    parser.add_argument("--train-positive-limit", type=int, default=2000)
    parser.add_argument("--train-negative-limit", type=int, default=1000)
    parser.add_argument("--valid-negative-limit", type=int, default=63)
    parser.add_argument("--test-negative-limit", type=int, default=204)
    parser.add_argument("--download-workers", type=int, default=12)
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> None:
    for rel in (
        "images/train",
        "images/valid",
        "images/test",
        "labels/train",
        "labels/valid",
        "labels/test",
        "manifests",
    ):
        (output_root / rel).mkdir(parents=True, exist_ok=True)


def cache_path(cache_root: Path, split: str, kind: str) -> Path:
    return cache_root / f"openimages_{split}_{kind}.csv"


def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        dest.write_bytes(response.read())
    return dest


def collect_positive_boxes(csv_path: Path, split: str, positive_limit: int | None) -> dict[str, list[tuple[float, float, float, float]]]:
    boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    kwargs = {"usecols": ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"]}
    if split == "train":
        reader = pd.read_csv(csv_path, chunksize=200000, **kwargs)
    else:
        reader = [pd.read_csv(csv_path, **kwargs)]

    for chunk in reader:
        matches = chunk[(chunk["LabelName"] == STREETLIGHT_LABEL_ID) & (chunk["IsGroupOf"] == 0) & (chunk["IsDepiction"] == 0)]
        for row in matches.itertuples(index=False):
            boxes[row.ImageID].append((float(row.XMin), float(row.YMin), float(row.XMax), float(row.YMax)))
        if positive_limit is not None and len(boxes) >= positive_limit:
            ordered = dict(sorted(boxes.items())[:positive_limit])
            return ordered
    return dict(sorted(boxes.items()))


def collect_negative_ids(csv_path: Path, split: str, positive_ids: set[str], negative_limit: int | None) -> list[str]:
    negatives: list[str] = []
    kwargs = {"usecols": ["ImageID", "LabelName", "Confidence"]}
    if split == "train":
        reader = pd.read_csv(csv_path, chunksize=200000, **kwargs)
    else:
        reader = [pd.read_csv(csv_path, **kwargs)]

    seen: set[str] = set()
    for chunk in reader:
        matches = chunk[(chunk["LabelName"] == STREETLIGHT_LABEL_ID) & (chunk["Confidence"] == 0)]
        for image_id in matches["ImageID"].tolist():
            if image_id in positive_ids or image_id in seen:
                continue
            negatives.append(image_id)
            seen.add(image_id)
            if negative_limit is not None and len(negatives) >= negative_limit:
                return negatives
    return negatives


def collect_image_index(csv_path: Path, needed_ids: set[str], split: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    kwargs = {
        "usecols": [
            "ImageID",
            "OriginalURL",
            "Thumbnail300KURL",
            "Rotation",
        ]
    }
    if split == "train":
        reader = pd.read_csv(csv_path, chunksize=200000, **kwargs)
    else:
        reader = [pd.read_csv(csv_path, **kwargs)]

    for chunk in reader:
        matches = chunk[chunk["ImageID"].isin(needed_ids)]
        for row in matches.to_dict("records"):
            rows[row["ImageID"]] = row
        if len(rows) == len(needed_ids):
            break
    return rows


def normalize_rotation(value: object) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_url(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def rotate_box_ccw(box: tuple[float, float, float, float], rotation: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = box
    rotation = rotation % 360
    if rotation == 0:
        return box
    if rotation == 90:
        return ymin, 1.0 - xmax, ymax, 1.0 - xmin
    if rotation == 180:
        return 1.0 - xmax, 1.0 - ymax, 1.0 - xmin, 1.0 - ymin
    if rotation == 270:
        return 1.0 - ymax, xmin, 1.0 - ymin, xmax
    raise ValueError(f"Unsupported rotation: {rotation}")


def build_records(
    split: str,
    positive_boxes: dict[str, list[tuple[float, float, float, float]]],
    negative_ids: list[str],
    image_index: dict[str, dict[str, str]],
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for image_id, boxes in positive_boxes.items():
        row = image_index.get(image_id)
        if not row:
            continue
        rotation = normalize_rotation(row.get("Rotation"))
        rotated_boxes = [rotate_box_ccw(box, rotation) for box in boxes]
        image_url = normalize_url(row.get("Thumbnail300KURL")) or normalize_url(row.get("OriginalURL"))
        if not image_url:
            continue
        records.append(
            SampleRecord(
                split=split,
                image_id=image_id,
                image_url=image_url,
                original_url=normalize_url(row.get("OriginalURL")) or image_url,
                rotation=rotation,
                boxes=rotated_boxes,
                is_negative=False,
            )
        )

    for image_id in negative_ids:
        row = image_index.get(image_id)
        if not row:
            continue
        image_url = normalize_url(row.get("Thumbnail300KURL")) or normalize_url(row.get("OriginalURL"))
        if not image_url:
            continue
        records.append(
            SampleRecord(
                split=split,
                image_id=image_id,
                image_url=image_url,
                original_url=normalize_url(row.get("OriginalURL")) or image_url,
                rotation=normalize_rotation(row.get("Rotation")),
                boxes=[],
                is_negative=True,
            )
        )

    return sorted(records, key=lambda item: (item.is_negative, item.image_id))


def yolo_line(box: tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    cx = xmin + (width / 2.0)
    cy = ymin + (height / 2.0)
    return f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def download_and_write_record(record: SampleRecord, output_root: Path) -> dict[str, str] | None:
    image_rel = Path("images") / record.split / f"{record.image_id}.jpg"
    label_rel = Path("labels") / record.split / f"{record.image_id}.txt"
    image_path = output_root / image_rel
    label_path = output_root / label_rel

    try:
        with urllib.request.urlopen(record.image_url, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError):
        return None

    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
        if record.rotation:
            image = image.rotate(record.rotation, expand=True)
        image.save(image_path, format="JPEG", quality=95)
    except Exception:
        return None

    lines = [yolo_line(box) for box in record.boxes]
    label_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "split": record.split,
        "image_id": record.image_id,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "source_url": record.image_url,
        "original_url": record.original_url,
        "rotation_ccw": str(record.rotation),
        "is_negative": "1" if record.is_negative else "0",
        "streetlight_count": str(len(record.boxes)),
        "source_dataset": "Open Images",
        "source_label_id": STREETLIGHT_LABEL_ID,
        "source_label_name": STREETLIGHT_LABEL_NAME,
    }


def write_outputs(output_root: Path, manifest_rows: list[dict[str, str]], summary: dict[str, object]) -> None:
    manifest_path = output_root / "manifests" / "openimages_streetlight_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "image_id",
                "image_path",
                "label_path",
                "source_url",
                "original_url",
                "rotation_ccw",
                "is_negative",
                "streetlight_count",
                "source_dataset",
                "source_label_id",
                "source_label_name",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

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
    (output_root / "manifests" / "openimages_streetlight_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_dirs(output_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    split_limits = {
        "train": {"positive": args.train_positive_limit, "negative": args.train_negative_limit},
        "valid": {"positive": None, "negative": args.valid_negative_limit},
        "test": {"positive": None, "negative": args.test_negative_limit},
    }

    all_records: list[SampleRecord] = []
    summary: dict[str, object] = {
        "source_dataset": "Open Images",
        "source_label_id": STREETLIGHT_LABEL_ID,
        "source_label_name": STREETLIGHT_LABEL_NAME,
        "policy": {
            "train_role": "external_training_augmentation",
            "valid_role": "held_out_external_validation",
            "test_role": "held_out_external_test",
            "image_download": "Thumbnail300KURL preferred, OriginalURL fallback",
        },
        "splits": {},
    }

    for split in ("train", "valid", "test"):
        bbox_csv = download_file(BOX_URLS[split], cache_path(cache_root, split, "bbox"))
        positive_boxes = collect_positive_boxes(bbox_csv, split, split_limits[split]["positive"])
        negative_csv = download_file(NEGATIVE_LABEL_URLS[split], cache_path(cache_root, split, "neglabels"))
        negative_ids = collect_negative_ids(negative_csv, split, set(positive_boxes.keys()), split_limits[split]["negative"])
        image_index_csv = download_file(IMAGE_INDEX_URLS[split], cache_path(cache_root, split, "images"))
        needed_ids = set(positive_boxes.keys()) | set(negative_ids)
        image_index = collect_image_index(image_index_csv, needed_ids, split)
        records = build_records(split, positive_boxes, negative_ids, image_index)
        all_records.extend(records)
        summary["splits"][split] = {
            "positive_images_selected": len(positive_boxes),
            "negative_images_selected": len(negative_ids),
            "positive_boxes_selected": int(sum(len(v) for v in positive_boxes.values())),
            "records_with_index_match": len(records),
        }

    manifest_rows: list[dict[str, str]] = []
    split_stats = {
        "train": {"images": 0, "positives": 0, "negatives": 0, "boxes": 0},
        "valid": {"images": 0, "positives": 0, "negatives": 0, "boxes": 0},
        "test": {"images": 0, "positives": 0, "negatives": 0, "boxes": 0},
    }

    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = [executor.submit(download_and_write_record, record, output_root) for record in all_records]
        for future in as_completed(futures):
            row = future.result()
            if not row:
                continue
            manifest_rows.append(row)
            split = row["split"]
            split_stats[split]["images"] += 1
            split_stats[split]["boxes"] += int(row["streetlight_count"])
            if row["is_negative"] == "1":
                split_stats[split]["negatives"] += 1
            else:
                split_stats[split]["positives"] += 1

    summary["materialized"] = split_stats
    write_outputs(output_root, sorted(manifest_rows, key=lambda item: (item["split"], item["image_id"])), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
