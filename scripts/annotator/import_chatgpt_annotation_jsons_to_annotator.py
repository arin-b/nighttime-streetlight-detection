from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any


ANNOTATOR_SCHEMA_VERSION = "measurement_annotator_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = ROOT / "exports" / "chatgpt batches"
DEFAULT_WORKSPACE = ROOT / "datasets" / "derived" / "modular_measurement_annotator" / "chatgpt_full_001_300_review" / "workspace"


def sanitize_key(value: str) -> str:
    cleaned = []
    for char in value.strip():
        cleaned.append(char if (char.isalnum() or char in "_.-") else "_")
    return "".join(cleaned).strip("._") or "item"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def normalize_dataset_zip(dataset_zip: str | None) -> str:
    if not dataset_zip:
        return ""
    return re.sub(r"\(\d+\)\.zip$", ".zip", dataset_zip)


def locate_batch_zip(batch_root: Path, dataset_zip: str) -> Path:
    exact = batch_root / dataset_zip
    if exact.exists():
        return exact
    matches = sorted(batch_root.glob(dataset_zip))
    if matches:
        return matches[0]
    matches = sorted(batch_root.glob(dataset_zip.replace(".zip", "*.zip")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate source zip for {dataset_zip} under {batch_root}")


def zip_member_name(zip_file: zipfile.ZipFile, image_name: str) -> str:
    suffix = f"/images/{image_name}"
    for member in zip_file.namelist():
        if member.endswith(suffix):
            return member
    raise FileNotFoundError(f"Could not locate {image_name} inside {zip_file.filename}")


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


def build_workspace(annotations_paths: list[Path], batch_root: Path, workspace: Path, force: bool) -> Path:
    if workspace.exists() and force:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    images_dir = workspace / "images"
    reviews_dir = workspace / "reviews" / "items"
    images_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for annotations_path in annotations_paths:
        payload = read_json(annotations_path)
        dataset_zip = normalize_dataset_zip(payload.get("dataset_zip"))
        source_zip = locate_batch_zip(batch_root, dataset_zip) if dataset_zip else None
        if source_zip is None:
            raise ValueError(f"Missing dataset_zip in {annotations_path}")

        with zipfile.ZipFile(source_zip, "r") as archive:
            for index, annotation in enumerate(payload.get("annotations", []), start=1):
                image_name = annotation["image_name"]
                member = zip_member_name(archive, image_name)
                out_name = image_name
                out_path = images_dir / out_name
                with archive.open(member, "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

                key = sanitize_key(f"{annotations_path.stem}_{index:03d}_{Path(image_name).stem}")
                width = int(annotation.get("width") or 0)
                height = int(annotation.get("height") or 0)
                review = {
                    "schema_version": ANNOTATOR_SCHEMA_VERSION,
                    "item_key": key,
                    "review_status": annotation.get("review_status", payload.get("review_status", "needs_review")),
                    "boxes": list(annotation.get("boxes") or []),
                    "confounder_boxes": list(annotation.get("confounder_boxes") or []),
                    "polygons": list(annotation.get("polygons") or []),
                    "measurement": normalize_measurement(annotation.get("measurement")),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "metadata": {
                        "source_annotation_file": str(annotations_path.resolve()),
                        "source_dataset_zip": dataset_zip,
                        "source_batch_zip": str(source_zip.resolve()),
                        "source_image_name": image_name,
                    },
                }
                write_json(reviews_dir / f"{key}.json", review)
                items.append(
                    {
                        "key": key,
                        "image_id": sanitize_key(Path(image_name).stem),
                        "image_path": str(out_path.resolve()),
                        "width": width,
                        "height": height,
                        "dataset_id": dataset_zip.replace(".zip", ""),
                        "route_id": "",
                        "clip_id": Path(dataset_zip).stem,
                        "frame_id": str(index),
                        "timestamp_ns": None,
                        "split": "review",
                        "source_pool": "chatgpt_draft",
                        "tracks": track_records(annotation.get("boxes") or []),
                        "metadata": {
                            "source_annotation_file": str(annotations_path.resolve()),
                            "source_dataset_zip": dataset_zip,
                            "source_batch_zip": str(source_zip.resolve()),
                            "source_image_name": image_name,
                        },
                    }
                )

    manifest = {
        "schema_version": ANNOTATOR_SCHEMA_VERSION,
        "workspace_id": sanitize_key(workspace.name),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_type": "chatgpt_visual_draft_import",
        "notes": "Combined import of uploaded ChatGPT annotation JSONs for human verification.",
        "items": items,
    }
    write_json(workspace / "manifest.json", manifest)

    with (workspace / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
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
            writer.writerow(
                {
                    "key": item["key"],
                    "image_id": item["image_id"],
                    "image_path": item["image_path"],
                    "dataset_id": item["dataset_id"],
                    "route_id": item["route_id"],
                    "clip_id": item["clip_id"],
                    "frame_id": item["frame_id"],
                    "timestamp_ns": item["timestamp_ns"] or "",
                    "width": item["width"],
                    "height": item["height"],
                    "split": item["split"],
                    "source_pool": item["source_pool"],
                    "track_count": len(item.get("tracks", [])),
                }
            )

    (workspace / "source_index.csv").write_text(
        "annotation_file,dataset_zip,source_batch_zip,item_count\n"
        + "\n".join(
            f"{row['annotation_file']},{row['dataset_zip']},{row['source_batch_zip']},{row['item_count']}"
            for row in (
                {
                    "annotation_file": str(path.resolve()),
                    "dataset_zip": normalize_dataset_zip(read_json(path).get("dataset_zip")),
                    "source_batch_zip": str(locate_batch_zip(batch_root, normalize_dataset_zip(read_json(path).get("dataset_zip"))).resolve()),
                    "item_count": len(read_json(path).get("annotations", [])),
                }
                for path in annotations_paths
            )
        ),
        encoding="utf-8",
    )

    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Import multiple ChatGPT annotation JSONs into one RBCCPS annotator workspace.")
    parser.add_argument("--annotations", nargs="+", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workspace = build_workspace(args.annotations, args.batch_root, args.workspace, args.force)
    print(workspace.resolve())


if __name__ == "__main__":
    main()
