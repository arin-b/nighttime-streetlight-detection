from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from rbccps_annotator.image_info import image_size
from rbccps_annotator.schema import (
    ANNOTATOR_SCHEMA_VERSION,
    IMAGE_EXTENSIONS,
    AnnotationItem,
    TrackRecord,
    WorkspaceManifest,
    default_review,
    normalize_lamp_class,
    sanitize_key,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_manifest(workspace: Path) -> dict[str, Any]:
    return read_json(workspace / "manifest.json", {})


def item_lookup(workspace: Path) -> dict[str, dict[str, Any]]:
    manifest = load_manifest(workspace)
    return {item["key"]: item for item in manifest.get("items", [])}


def review_path(workspace: Path, key: str) -> Path:
    return workspace / "reviews" / "items" / f"{sanitize_key(key)}.json"


def load_review(workspace: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = review_path(workspace, item["key"])
    if path.exists():
        return _normalize_review(read_json(path, {}), item["key"])
    typed_item = AnnotationItem(
        key=item["key"],
        image_id=item["image_id"],
        image_path=item["image_path"],
        width=int(item.get("width", 0)),
        height=int(item.get("height", 0)),
        tracks=[TrackRecord(**track) for track in item.get("tracks", [])],
    )
    return default_review(typed_item)


def _normalize_review(review: dict[str, Any], key: str) -> dict[str, Any]:
    review.setdefault("schema_version", ANNOTATOR_SCHEMA_VERSION)
    review.setdefault("item_key", key)
    review.setdefault("review_status", "unreviewed")
    review.setdefault("boxes", [])
    for box in review["boxes"]:
        box["class_name"] = normalize_lamp_class(box.get("class_name"))
        box.setdefault("parent_pole_box_id", "")
        box.setdefault("notes", "")
    review.setdefault("confounder_boxes", [])
    review.setdefault("polygons", [])
    measurement = review.setdefault("measurement", {})
    measurement.setdefault("lamp_status", [])
    measurement.setdefault("public_space_regions", [])
    measurement.setdefault("affected_regions", [])
    measurement.setdefault("visibility_labels", [])
    measurement.setdefault("attribution_labels", [])
    measurement.setdefault("lux_points", [])
    measurement.setdefault("qa_flags", [])
    review.setdefault("updated_at", "")
    return review


def save_review(workspace: Path, key: str, review: dict[str, Any]) -> dict[str, Any]:
    if key not in item_lookup(workspace):
        raise ValueError(f"Unknown item key: {key}")
    review["schema_version"] = ANNOTATOR_SCHEMA_VERSION
    review["item_key"] = key
    review["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(review_path(workspace, key), review)
    return review


def create_workspace_from_frames(
    frames: Path,
    workspace: Path,
    dataset_id: str,
    route_id: str,
    clip_id: str,
    split: str,
    source_pool: str,
) -> Path:
    frame_paths = sorted(path for path in frames.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not frame_paths:
        raise ValueError(f"No supported images found under {frames}")
    ensure_dir(workspace)
    items: list[AnnotationItem] = []
    for index, image_path in enumerate(frame_paths, start=1):
        width, height = image_size(image_path)
        image_id = sanitize_key(image_path.stem)
        key = sanitize_key(f"{dataset_id}_{clip_id or frames.name}_{index:06d}_{image_id}")
        items.append(
            AnnotationItem(
                key=key,
                image_id=image_id,
                image_path=str(image_path.resolve()),
                width=width,
                height=height,
                dataset_id=dataset_id,
                route_id=route_id,
                clip_id=clip_id or frames.name,
                frame_id=str(index),
                split=split,
                source_pool=source_pool,
            )
        )
    manifest = WorkspaceManifest(
        schema_version=ANNOTATOR_SCHEMA_VERSION,
        workspace_id=sanitize_key(workspace.name),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_type="raw_frames",
        items=items,
    )
    write_json(workspace / "manifest.json", manifest.to_dict())
    _write_frames_csv(workspace, items)
    return workspace


def create_workspace_from_detector_manifest(manifest_path: Path, workspace: Path) -> Path:
    payload = read_json(manifest_path, {})
    frames_by_id: dict[int, dict[str, Any]] = {}
    tracks_by_frame: dict[int, list[TrackRecord]] = {}
    for frame in payload.get("frames", []):
        frame_id = int(frame["frame_id"])
        frames_by_id[frame_id] = frame
    for track in payload.get("tracks", []):
        frame_id = int(track["frame_id"])
        tracks_by_frame.setdefault(frame_id, []).append(
            TrackRecord(
                track_id=str(track["track_id"]),
                bbox_xyxy=[float(value) for value in track["bbox_xyxy"]],
                class_name=normalize_lamp_class(track.get("class_name")),
                detector_score=track.get("detector_score"),
                track_confidence=track.get("track_confidence"),
                track_age=track.get("track_age"),
                lost_count=track.get("lost_count"),
                source_model=track.get("source_model"),
            )
        )
    ensure_dir(workspace)
    items: list[AnnotationItem] = []
    base = manifest_path.parent
    for frame_id in sorted(frames_by_id):
        frame = frames_by_id[frame_id]
        image_uri = str(frame["image_uri"])
        image_path = Path(image_uri)
        if not image_path.is_absolute():
            image_path = (base / image_path).resolve()
        width = int(frame.get("width") or 0)
        height = int(frame.get("height") or 0)
        if (width == 0 or height == 0) and image_path.exists():
            width, height = image_size(image_path)
        key = sanitize_key(f"{payload.get('clip_id', 'clip')}_{frame_id}")
        items.append(
            AnnotationItem(
                key=key,
                image_id=key,
                image_path=str(image_path),
                width=width,
                height=height,
                dataset_id=str(payload.get("device_id", "detector_run")),
                route_id=str(payload.get("route_id", "")),
                clip_id=str(payload.get("clip_id", manifest_path.stem)),
                frame_id=str(frame_id),
                timestamp_ns=int(frame.get("timestamp_ns", 0)),
                source_pool="detector_run",
                tracks=tracks_by_frame.get(frame_id, []),
                metadata={
                    "camera": frame.get("camera", {}),
                    "pose": frame.get("pose", {}),
                    "calibration_level": payload.get("calibration_level"),
                    "policy_id": payload.get("policy_id"),
                    "video_uri": payload.get("video_uri"),
                },
            )
        )
    manifest = WorkspaceManifest(
        schema_version=ANNOTATOR_SCHEMA_VERSION,
        workspace_id=sanitize_key(workspace.name),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_type="detector_run",
        items=items,
        notes=f"Imported from {manifest_path}",
    )
    write_json(workspace / "manifest.json", manifest.to_dict())
    _write_frames_csv(workspace, items)
    return workspace


def _write_frames_csv(workspace: Path, items: list[AnnotationItem]) -> None:
    rows = []
    for item in items:
        rows.append(
            {
                "key": item.key,
                "image_id": item.image_id,
                "image_path": item.image_path,
                "dataset_id": item.dataset_id,
                "route_id": item.route_id,
                "clip_id": item.clip_id,
                "frame_id": item.frame_id,
                "timestamp_ns": item.timestamp_ns or "",
                "width": item.width,
                "height": item.height,
                "split": item.split,
                "source_pool": item.source_pool,
                "track_count": len(item.tracks),
            }
        )
    write_csv(
        workspace / "frames.csv",
        rows,
        [
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
        ],
    )
