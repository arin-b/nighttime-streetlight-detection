from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any


ANNOTATOR_SCHEMA_VERSION = "measurement_annotator_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = ROOT / "exports" / "chatgpt_annotation_batch_100" / "chatgpt_streetlight_measurement_batch_100"
DEFAULT_ANNOTATIONS = ROOT / "streetlight_annotations_batch_001_020.json"
DEFAULT_WORKSPACE = ROOT / "datasets" / "derived" / "modular_measurement_annotator" / "chatgpt_batch_001_020_review" / "workspace"


def sanitize_key(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in "_.-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    key = "".join(cleaned).strip("._")
    return key or "item"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_batch_manifest(batch_root: Path) -> dict[str, dict[str, str]]:
    manifest_csv = batch_root / "image_manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[row["image_name"]] = row
    return rows


def normalize_measurement(measurement: dict[str, Any] | None) -> dict[str, list[Any]]:
    measurement = measurement or {}
    return {
        "lamp_status": list(measurement.get("lamp_status") or []),
        "public_space_regions": list(measurement.get("public_space_regions") or []),
        "affected_regions": list(measurement.get("affected_regions") or []),
        "visibility_labels": list(measurement.get("visibility_labels") or []),
        "attribution_labels": list(measurement.get("attribution_labels") or []),
        "lux_points": list(measurement.get("lux_points") or []),
        "qa_flags": list(measurement.get("qa_flags") or []),
    }


def track_records(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tracks = []
    for index, box in enumerate(boxes, start=1):
        track_id = str(box.get("track_id") or f"track_{index:03d}")
        tracks.append(
            {
                "track_id": track_id,
                "bbox_xyxy": [float(value) for value in box.get("bbox_xyxy", [0, 0, 0, 0])],
                "class_name": str(box.get("class_name") or "streetlight_lamp_head"),
                "detector_score": None,
                "track_confidence": None,
                "track_age": None,
                "lost_count": None,
                "source_model": "chatgpt_visual_draft",
            }
        )
    return tracks


def import_annotations(annotations_path: Path, batch_root: Path, workspace: Path, force: bool) -> Path:
    payload = read_json(annotations_path)
    batch_rows = load_batch_manifest(batch_root)
    images_dir = batch_root / "images"

    if workspace.exists() and force:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    items = []
    for index, annotation in enumerate(payload.get("annotations", []), start=1):
        image_name = annotation["image_name"]
        batch_row = batch_rows.get(image_name)
        if not batch_row:
            raise ValueError(f"Image not found in batch manifest: {image_name}")
        image_path = (images_dir / image_name).resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        key = sanitize_key(f"chatgpt_batch_001_020_{index:03d}_{Path(image_name).stem}")
        width = int(annotation.get("width") or batch_row["width"])
        height = int(annotation.get("height") or batch_row["height"])
        boxes = list(annotation.get("boxes") or [])

        items.append(
            {
                "key": key,
                "image_id": sanitize_key(Path(image_name).stem),
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "dataset_id": "chatgpt_streetlight_measurement_batch_100",
                "route_id": str(batch_row.get("source_group", "")),
                "clip_id": str(batch_row.get("source_group", "")),
                "frame_id": str(batch_row.get("batch_index", index)),
                "timestamp_ns": None,
                "split": "review",
                "source_pool": str(batch_row.get("source_pool", "chatgpt_batch")),
                "tracks": track_records(boxes),
                "metadata": {
                    "source_annotation_file": str(annotations_path.resolve()),
                    "source_image_name": image_name,
                    "batch_row": batch_row,
                },
            }
        )

        review = {
            "schema_version": ANNOTATOR_SCHEMA_VERSION,
            "item_key": key,
            "review_status": annotation.get("review_status", payload.get("review_status", "needs_review")),
            "boxes": boxes,
            "confounder_boxes": list(annotation.get("confounder_boxes") or []),
            "polygons": list(annotation.get("polygons") or []),
            "measurement": normalize_measurement(annotation.get("measurement")),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": {
                "imported_from": "chatgpt_visual_draft",
                "source_image_name": image_name,
            },
        }
        write_json(workspace / "reviews" / "items" / f"{sanitize_key(key)}.json", review)

    manifest = {
        "schema_version": ANNOTATOR_SCHEMA_VERSION,
        "workspace_id": sanitize_key(workspace.name),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_type": "chatgpt_visual_draft_import",
        "notes": f"Imported from {annotations_path.resolve()} for visual review in the RBCCPS annotator.",
        "items": items,
    }
    write_json(workspace / "manifest.json", manifest)

    frames_csv = workspace / "frames.csv"
    with frames_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "key",
            "image_id",
            "image_path",
            "dataset_id",
            "route_id",
            "clip_id",
            "frame_id",
            "timestamp_ns",
            "width",
            "height",
            "split",
            "source_pool",
            "track_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field, "") for field in fieldnames[:-1]} | {"track_count": len(item.get("tracks", []))})

    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Import ChatGPT draft streetlight measurement annotations into the RBCCPS annotator workspace format.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workspace = import_annotations(args.annotations, args.batch_root, args.workspace, args.force)
    print(workspace.resolve())


if __name__ == "__main__":
    main()
