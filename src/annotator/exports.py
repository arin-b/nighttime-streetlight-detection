from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from rbccps_annotator.schema import LAMP_BOX_CLASSES, normalize_lamp_class
from rbccps_annotator.workspace import ensure_dir, load_manifest, load_review, write_csv, write_json

YOLO_CLASS_IDS = {class_name: index for index, class_name in enumerate(LAMP_BOX_CLASSES)}


def export_yolo(workspace: Path, output: Path, include_candidate_boxes: bool = False, split_dirs: bool = False) -> Path:
    manifest = load_manifest(workspace)
    labels_dir = ensure_dir(output / "labels")
    images_dir = ensure_dir(output / "images")
    if split_dirs:
        for split_name in ("train", "valid", "test"):
            ensure_dir(images_dir / split_name)
            ensure_dir(labels_dir / split_name)
    sidecar_rows: list[dict[str, Any]] = []
    for item in manifest.get("items", []):
        review = load_review(workspace, item)
        width = float(item.get("width") or 0)
        height = float(item.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        source = Path(item["image_path"])
        image_name = f"{item['key']}{source.suffix.lower()}"
        split = str(item.get("split") or "train")
        if split_dirs:
            item_images_dir = ensure_dir(images_dir / split)
            item_labels_dir = ensure_dir(labels_dir / split)
            train_path = "images/train"
            val_path = "images/valid"
            test_path = "images/test"
        else:
            item_images_dir = images_dir
            item_labels_dir = labels_dir
            train_path = "images"
            val_path = "images"
            test_path = "images"
        if source.exists():
            shutil.copy2(source, item_images_dir / image_name)
        label_lines = []
        for box in review.get("boxes", []):
            status = str(box.get("status", "candidate"))
            if status in {"false_positive", "exclude", "deleted"}:
                continue
            if status == "candidate" and not include_candidate_boxes:
                continue
            class_name = normalize_lamp_class(box.get("class_name"))
            class_id = YOLO_CLASS_IDS.get(class_name)
            if class_id is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box["bbox_xyxy"]]
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            bw = max(0.0, x2 - x1) / width
            bh = max(0.0, y2 - y1) / height
            label_lines.append(f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
            sidecar_rows.append(
                {
                    "key": item["key"],
                    "image_path": str(item_images_dir / image_name),
                    "label_path": str(item_labels_dir / f"{item['key']}.txt"),
                    "split": split,
                    "box_id": box.get("box_id", ""),
                    "class_name": class_name,
                    "track_id": box.get("track_id", ""),
                    "parent_pole_box_id": box.get("parent_pole_box_id", ""),
                    "status": status,
                    "source": box.get("source", ""),
                }
            )
        (item_labels_dir / f"{item['key']}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(LAMP_BOX_CLASSES) + "\n", encoding="utf-8")
    write_csv(output / "yolo_sidecar.csv", sidecar_rows, ["key", "image_path", "label_path", "split", "box_id", "class_name", "track_id", "parent_pole_box_id", "status", "source"])
    write_json(output / "dataset.yaml", {"path": str(output.resolve()), "train": train_path, "val": val_path, "test": test_path, "names": {index: name for index, name in enumerate(LAMP_BOX_CLASSES)}})
    return output


def export_measurement(workspace: Path, output: Path) -> Path:
    manifest = load_manifest(workspace)
    ensure_dir(output)
    track_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    visibility_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    lux_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    polygon_rows: list[dict[str, Any]] = []
    confounder_box_rows: list[dict[str, Any]] = []
    affected_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []

    for item in manifest.get("items", []):
        review = load_review(workspace, item)
        base = _base_row(item)
        for box in review.get("boxes", []):
            track_rows.append(
                base
                | {
                    "box_id": box.get("box_id", ""),
                    "class_name": normalize_lamp_class(box.get("class_name")),
                    "track_id": box.get("track_id", ""),
                    "parent_pole_box_id": box.get("parent_pole_box_id", ""),
                    "bbox_xyxy": json.dumps(box.get("bbox_xyxy", [])),
                    "box_status": box.get("status", ""),
                }
            )
        for box in review.get("confounder_boxes", []):
            confounder_box_rows.append(
                base
                | {
                    "box_id": box.get("box_id", ""),
                    "surface_type": box.get("surface_type", ""),
                    "bbox_xyxy": json.dumps(box.get("bbox_xyxy", [])),
                    "source": box.get("source", ""),
                    "notes": box.get("notes", ""),
                    "can_confound_streetlight": box.get("can_confound_streetlight", True),
                    "augmentation_allowed": box.get("augmentation_allowed", False),
                }
            )
        for polygon in review.get("polygons", []):
            polygon_rows.append(base | {"polygon_id": polygon.get("polygon_id", ""), "surface_type": polygon.get("surface_type", ""), "points_json": json.dumps(polygon.get("points", [])), "augmentation_allowed": polygon.get("augmentation_allowed", False), "can_confound_streetlight": polygon.get("can_confound_streetlight", False)})
        measurement = review.get("measurement", {})
        for row in measurement.get("lamp_status", []):
            status_rows.append(base | row)
        for row in measurement.get("public_space_regions", []):
            public_rows.append(base | _json_points(row))
        for row in measurement.get("affected_regions", []):
            affected_rows.append(base | _json_points(row))
        for row in measurement.get("visibility_labels", []):
            visibility_rows.append(base | row)
        for row in measurement.get("attribution_labels", []):
            attribution_rows.append(base | row)
        for row in measurement.get("lux_points", []):
            lux_rows.append(base | row)
        for row in measurement.get("qa_flags", []):
            qa_rows.append(base | row)

    write_csv(output / "tracks.csv", track_rows, _fieldnames(track_rows))
    write_csv(output / "confounder_boxes.csv", confounder_box_rows, _fieldnames(confounder_box_rows))
    write_csv(output / "confounder_polygons.csv", polygon_rows, _fieldnames(polygon_rows))
    write_csv(output / "lamp_status.csv", status_rows, _fieldnames(status_rows))
    write_csv(output / "public_space_regions.csv", public_rows, _fieldnames(public_rows))
    write_csv(output / "affected_regions.csv", affected_rows, _fieldnames(affected_rows))
    write_csv(output / "visibility_labels.csv", visibility_rows, _fieldnames(visibility_rows))
    write_csv(output / "attribution_labels.csv", attribution_rows, _fieldnames(attribution_rows))
    write_csv(output / "lux_points.csv", lux_rows, _fieldnames(lux_rows))
    write_csv(output / "qa_flags.csv", qa_rows, _fieldnames(qa_rows))
    write_json(output / "measurement_annotation_manifest.json", {"source_workspace": str(workspace.resolve()), "item_count": len(manifest.get("items", []))})
    return output


def _base_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": item.get("key", ""),
        "route_id": item.get("route_id", ""),
        "clip_id": item.get("clip_id", ""),
        "frame_id": item.get("frame_id", ""),
        "timestamp_ns": item.get("timestamp_ns", ""),
        "image_path": item.get("image_path", ""),
        "width": item.get("width", ""),
        "height": item.get("height", ""),
    }


def _json_points(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    if "points" in payload:
        payload["points_json"] = json.dumps(payload.pop("points"))
    return payload


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["key"]
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names
