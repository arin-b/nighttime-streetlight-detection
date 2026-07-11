from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.dataset_prep.normalization import (
    NormalizedValue,
    normalize_affected_region,
    normalize_attribution,
    normalize_confounder,
    normalize_lamp_status,
    normalize_public_region,
    normalize_visibility,
)
from rbccps_measurement.ingest.validation import validate_clip_manifest


VALIDATION_MODES = {"warn", "quarantine", "fail"}
INPUT_TYPES = {"llm-json", "annotator-workspace", "measurement-export"}
BASE_COLUMNS = [
    "source_file",
    "source_item",
    "image_name",
    "clip_id",
    "frame_id",
    "track_id",
    "raw_label",
    "normalized_label",
    "validation_status",
]


@dataclass
class Issue:
    severity: str
    table: str
    source_file: str
    source_item: str
    clip_id: str
    frame_id: str
    track_id: str
    field: str
    raw_value: str
    normalized_value: str
    message: str

    def to_row(self) -> dict[str, str]:
        return self.__dict__.copy()


@dataclass
class PreparedData:
    frames: list[dict[str, Any]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    lamp_status: list[dict[str, Any]] = field(default_factory=list)
    public_space_regions: list[dict[str, Any]] = field(default_factory=list)
    affected_regions: list[dict[str, Any]] = field(default_factory=list)
    confounders: list[dict[str, Any]] = field(default_factory=list)
    visibility_labels: list[dict[str, Any]] = field(default_factory=list)
    attribution_labels: list[dict[str, Any]] = field(default_factory=list)
    qa_flags: list[dict[str, Any]] = field(default_factory=list)
    lux_points: list[dict[str, Any]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


def convert_annotations(input_path: str | Path, input_type: str, out: str | Path, validation_mode: str = "warn") -> PreparedData:
    input_path = Path(input_path)
    out = Path(out)
    if input_type not in INPUT_TYPES:
        raise ValueError(f"unsupported input_type {input_type!r}; expected one of {sorted(INPUT_TYPES)}")
    if validation_mode not in VALIDATION_MODES:
        raise ValueError(f"unsupported validation_mode {validation_mode!r}; expected one of {sorted(VALIDATION_MODES)}")

    if input_type == "llm-json":
        data = _load_llm_json(input_path)
    elif input_type == "annotator-workspace":
        data = _load_annotator_workspace(input_path)
    else:
        data = _load_measurement_export(input_path)

    _validate_references(data)
    if validation_mode == "fail" and data.issues:
        first = data.issues[0]
        raise ValueError(f"{first.table}.{first.field}: {first.message}")

    _write_prepared_dataset(data, out, validation_mode)
    return data


def _source_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob(pattern))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _base(source_file: Path, source_item: str, image_name: str, clip_id: str, frame_id: str, track_id: str = "") -> dict[str, Any]:
    return {
        "source_file": str(source_file),
        "source_item": source_item,
        "image_name": image_name,
        "clip_id": clip_id,
        "frame_id": str(frame_id),
        "track_id": str(track_id or ""),
        "raw_label": "",
        "normalized_label": "",
        "validation_status": "valid",
    }


def _record_issue(
    data: PreparedData,
    row: dict[str, Any],
    table: str,
    field: str,
    raw_value: object,
    normalized_value: object,
    message: str,
    severity: str = "warning",
) -> None:
    row["validation_status"] = "warning" if row.get("validation_status") == "valid" else row.get("validation_status", "warning")
    data.issues.append(
        Issue(
            severity=severity,
            table=table,
            source_file=str(row.get("source_file", "")),
            source_item=str(row.get("source_item", "")),
            clip_id=str(row.get("clip_id", "")),
            frame_id=str(row.get("frame_id", "")),
            track_id=str(row.get("track_id", "")),
            field=field,
            raw_value=str(raw_value if raw_value is not None else ""),
            normalized_value=str(normalized_value if normalized_value is not None else ""),
            message=message,
        )
    )


def _apply_normalized(data: PreparedData, row: dict[str, Any], table: str, field: str, value: NormalizedValue) -> None:
    row["raw_label"] = value.raw
    row["normalized_label"] = value.normalized
    if value.warning:
        _record_issue(data, row, table, field, value.raw, value.normalized, value.warning)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in {None, ""} else default
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value not in {None, ""} else default
    except (TypeError, ValueError):
        return default


def _points_json(row: dict[str, Any]) -> str:
    if "points_json" in row and row["points_json"] not in {None, ""}:
        return str(row["points_json"])
    return json.dumps(row.get("points", []))


def _clip_bbox(data: PreparedData, row: dict[str, Any], table: str, bbox: Iterable[object], width: int, height: int) -> list[float]:
    values = [_to_float(value) for value in list(bbox)[:4]]
    if len(values) != 4:
        values = [0.0, 0.0, 1.0, 1.0]
        _record_issue(data, row, table, "bbox_xyxy", bbox, values, "bbox must contain four values")
    x1, y1, x2, y2 = values
    clipped = [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    if clipped != values:
        _record_issue(data, row, table, "bbox_xyxy", values, clipped, "bbox outside frame bounds; clipped for prepared manifest")
    if clipped[2] <= clipped[0]:
        clipped[2] = min(float(width), clipped[0] + 1.0)
        _record_issue(data, row, table, "bbox_xyxy", values, clipped, "bbox width was non-positive; widened to one pixel")
    if clipped[3] <= clipped[1]:
        clipped[3] = min(float(height), clipped[1] + 1.0)
        _record_issue(data, row, table, "bbox_xyxy", values, clipped, "bbox height was non-positive; heightened to one pixel")
    return [round(value, 4) for value in clipped]


def _frame_row(source_file: Path, source_item: str, image_name: str, clip_id: str, frame_id: str, width: int, height: int, timestamp_ns: int | None = None) -> dict[str, Any]:
    row = _base(source_file, source_item, image_name, clip_id, frame_id)
    row.update(
        {
            "width": width,
            "height": height,
            "timestamp_ns": timestamp_ns if timestamp_ns is not None else 1_700_000_000_000_000_000 + (_to_int(frame_id, 1) - 1) * 33_333_333,
            "image_path": image_name,
            "metadata_quality": "pseudo",
            "calibration_level": 1,
            "policy_id": "rbccps_measurement_policy_v1",
        }
    )
    return row


def _measurement_dict(payload: dict[str, Any]) -> dict[str, list[Any]]:
    measurement = payload.get("measurement") or {}
    return {
        "lamp_status": list(measurement.get("lamp_status") or []),
        "public_space_regions": list(measurement.get("public_space_regions") or []),
        "affected_regions": list(measurement.get("affected_regions") or []),
        "visibility_labels": list(measurement.get("visibility_labels") or []),
        "attribution_labels": list(measurement.get("attribution_labels") or []),
        "lux_points": list(measurement.get("lux_points") or []),
        "qa_flags": list(measurement.get("qa_flags") or []),
    }


def _load_llm_json(path: Path) -> PreparedData:
    data = PreparedData()
    for source_file in _source_files(path, "annotation_llm.json"):
        payload = _read_json(source_file)
        clip_id = Path(str(payload.get("dataset_zip") or source_file.stem)).stem
        for index, annotation in enumerate(payload.get("annotations", []), start=1):
            image_name = str(annotation.get("image_name") or f"frame_{index:06d}.jpg")
            frame_id = str(index)
            width = _to_int(annotation.get("width"), 1)
            height = _to_int(annotation.get("height"), 1)
            source_item = f"annotations[{index - 1}]"
            data.frames.append(_frame_row(source_file, source_item, image_name, clip_id, frame_id, width, height))
            _append_boxes(data, source_file, source_item, image_name, clip_id, frame_id, annotation.get("boxes") or [], width, height)
            _append_confounders(data, source_file, source_item, image_name, clip_id, frame_id, annotation.get("confounder_boxes") or [], width, height)
            _append_measurement(data, source_file, source_item, image_name, clip_id, frame_id, _measurement_dict(annotation))
    return data


def _load_annotator_workspace(path: Path) -> PreparedData:
    data = PreparedData()
    manifest = _read_json(path / "manifest.json")
    for item in manifest.get("items", []):
        key = str(item["key"])
        review_path = path / "reviews" / "items" / f"{key}.json"
        review = _read_json(review_path) if review_path.exists() else {}
        source_file = review_path if review_path.exists() else path / "manifest.json"
        image_name = Path(str(item.get("image_path") or key)).name
        clip_id = str(item.get("clip_id") or manifest.get("workspace_id") or "annotator_workspace")
        frame_id = str(item.get("frame_id") or key)
        width = _to_int(item.get("width"), 1)
        height = _to_int(item.get("height"), 1)
        timestamp_ns = _to_int(item.get("timestamp_ns"), 0) or None
        data.frames.append(_frame_row(source_file, key, image_name, clip_id, frame_id, width, height, timestamp_ns))
        _append_boxes(data, source_file, key, image_name, clip_id, frame_id, review.get("boxes") or item.get("tracks") or [], width, height)
        _append_confounders(data, source_file, key, image_name, clip_id, frame_id, review.get("confounder_boxes") or [], width, height)
        _append_measurement(data, source_file, key, image_name, clip_id, frame_id, _measurement_dict(review))
    return data


def _load_measurement_export(path: Path) -> PreparedData:
    data = PreparedData()
    csvs = {csv_path.stem: _read_csv(csv_path) for csv_path in path.glob("*.csv")}
    frames_by_key: dict[str, dict[str, Any]] = {}
    for row in csvs.get("tracks", []) + csvs.get("lamp_status", []) + csvs.get("public_space_regions", []):
        key = str(row.get("key") or f"{row.get('clip_id', 'clip')}_{row.get('frame_id', 'frame')}")
        if key not in frames_by_key:
            source = path / "measurement_annotation_manifest.json"
            image_name = Path(str(row.get("image_path") or key)).name
            frame = _frame_row(
                source,
                key,
                image_name,
                str(row.get("clip_id") or "measurement_export"),
                str(row.get("frame_id") or key),
                _to_int(row.get("width"), 1),
                _to_int(row.get("height"), 1),
                _to_int(row.get("timestamp_ns"), 0) or None,
            )
            frames_by_key[key] = frame
            data.frames.append(frame)
    frame_lookup = {(row["clip_id"], row["frame_id"]): row for row in data.frames}
    for row in csvs.get("tracks", []):
        frame = frame_lookup.get((str(row.get("clip_id") or "measurement_export"), str(row.get("frame_id") or row.get("key"))))
        if frame:
            _append_boxes(data, path / "tracks.csv", str(row.get("key", "")), frame["image_name"], frame["clip_id"], frame["frame_id"], [row], _to_int(frame["width"], 1), _to_int(frame["height"], 1))
    for row in csvs.get("confounder_boxes", []) + csvs.get("confounder_polygons", []):
        frame = _find_frame(data.frames, row)
        if frame:
            _append_confounders(data, path / "confounders.csv", str(row.get("key", "")), frame["image_name"], frame["clip_id"], frame["frame_id"], [row], _to_int(frame["width"], 1), _to_int(frame["height"], 1))
    measurement = {
        "lamp_status": csvs.get("lamp_status", []),
        "public_space_regions": csvs.get("public_space_regions", []),
        "affected_regions": csvs.get("affected_regions", []),
        "visibility_labels": csvs.get("visibility_labels", []),
        "attribution_labels": csvs.get("attribution_labels", []),
        "lux_points": csvs.get("lux_points", []),
        "qa_flags": csvs.get("qa_flags", []),
    }
    for table, rows in measurement.items():
        for row in rows:
            frame = _find_frame(data.frames, row)
            if frame:
                _append_measurement(data, path / f"{table}.csv", str(row.get("key", "")), frame["image_name"], frame["clip_id"], frame["frame_id"], {table: [row]})
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _find_frame(frames: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    clip_id = str(row.get("clip_id") or "")
    frame_id = str(row.get("frame_id") or "")
    key = str(row.get("key") or "")
    for frame in frames:
        if (clip_id and frame["clip_id"] == clip_id and frame["frame_id"] == frame_id) or frame["source_item"] == key:
            return frame
    return None


def _append_boxes(data: PreparedData, source_file: Path, source_item: str, image_name: str, clip_id: str, frame_id: str, boxes: list[dict[str, Any]], width: int, height: int) -> None:
    for index, box in enumerate(boxes, start=1):
        track_id = str(box.get("track_id") or f"track_{index:03d}")
        row = _base(source_file, source_item, image_name, clip_id, frame_id, track_id)
        raw_bbox = box.get("bbox_xyxy", [])
        if isinstance(raw_bbox, str):
            try:
                raw_bbox = json.loads(raw_bbox)
            except json.JSONDecodeError:
                raw_bbox = []
        clipped = _clip_bbox(data, row, "tracks", raw_bbox, width, height)
        class_name = str(box.get("class_name") or "streetlight_lamp_head")
        if class_name in {"", "streetlight"}:
            class_name = "streetlight_lamp_head"
        row.update(
            {
                "box_id": box.get("box_id", ""),
                "class_name": class_name,
                "bbox_xyxy": json.dumps(clipped),
                "raw_bbox_xyxy": json.dumps(raw_bbox),
                "detector_score": box.get("detector_score") or box.get("confidence") or 0.5,
                "track_confidence": box.get("track_confidence") or "",
                "track_age": box.get("track_age") or "",
                "lost_count": box.get("lost_count") or 0,
                "source_model": box.get("source_model") or box.get("source") or "annotation_converter",
            }
        )
        data.tracks.append(row)


def _append_confounders(data: PreparedData, source_file: Path, source_item: str, image_name: str, clip_id: str, frame_id: str, rows: list[dict[str, Any]], width: int, height: int) -> None:
    for index, item in enumerate(rows, start=1):
        row = _base(source_file, source_item, image_name, clip_id, frame_id, str(item.get("track_id") or ""))
        raw_label = item.get("surface_type") or item.get("source_type") or item.get("region_type") or item.get("class_name") or ""
        normalized = normalize_confounder(raw_label)
        _apply_normalized(data, row, "confounders", "surface_type", normalized)
        bbox = item.get("bbox_xyxy")
        if isinstance(bbox, str):
            try:
                bbox = json.loads(bbox)
            except json.JSONDecodeError:
                bbox = None
        row.update(
            {
                "confounder_id": item.get("box_id") or item.get("polygon_id") or f"confounder_{index:03d}",
                "confounder_type": normalized.normalized,
                "raw_confounder_type": normalized.raw,
                "bbox_xyxy": json.dumps(_clip_bbox(data, row, "confounders", bbox, width, height)) if bbox else "",
                "points_json": item.get("points_json") or (json.dumps(item.get("points", [])) if item.get("points") else ""),
                "can_confound_streetlight": item.get("can_confound_streetlight", True),
                "notes": item.get("notes", ""),
            }
        )
        data.confounders.append(row)


def _append_measurement(data: PreparedData, source_file: Path, source_item: str, image_name: str, clip_id: str, frame_id: str, measurement: dict[str, list[Any]]) -> None:
    for item in measurement.get("lamp_status", []):
        track_id = str(item.get("track_id") or "")
        row = _base(source_file, source_item, image_name, clip_id, frame_id, track_id)
        normalized = normalize_lamp_status(item.get("status"))
        _apply_normalized(data, row, "lamp_status", "status", normalized)
        row.update({"status": normalized.normalized, "notes": item.get("notes", "")})
        data.lamp_status.append(row)
    for item in measurement.get("public_space_regions", []):
        row = _base(source_file, source_item, image_name, clip_id, frame_id, str(item.get("track_id") or ""))
        normalized = normalize_public_region(item.get("region_type"))
        _apply_normalized(data, row, "public_space_regions", "region_type", normalized)
        row.update({"region_type": normalized.normalized, "points_json": _points_json(item), "notes": item.get("notes", "")})
        data.public_space_regions.append(row)
    for item in measurement.get("affected_regions", []):
        row = _base(source_file, source_item, image_name, clip_id, frame_id, str(item.get("track_id") or ""))
        normalized = normalize_affected_region(item.get("region_type"))
        _apply_normalized(data, row, "affected_regions", "region_type", normalized)
        row.update({"region_type": normalized.normalized, "points_json": _points_json(item), "visibility_quality": item.get("visibility_quality", ""), "notes": item.get("notes", "")})
        data.affected_regions.append(row)
    for item in measurement.get("visibility_labels", []):
        row = _base(source_file, source_item, image_name, clip_id, frame_id, str(item.get("track_id") or ""))
        normalized = normalize_visibility(item.get("visibility_class"))
        _apply_normalized(data, row, "visibility_labels", "visibility_class", normalized)
        row.update({"visibility_class": normalized.normalized, "notes": item.get("notes", "")})
        data.visibility_labels.append(row)
    for item in measurement.get("attribution_labels", []):
        row = _base(source_file, source_item, image_name, clip_id, frame_id, str(item.get("track_id") or ""))
        normalized = normalize_attribution(item.get("attribution_class"))
        _apply_normalized(data, row, "attribution_labels", "attribution_class", normalized)
        row.update({"attribution_class": normalized.normalized, "evidence": item.get("evidence", ""), "notes": item.get("notes", "")})
        data.attribution_labels.append(row)
    for item in measurement.get("lux_points", []):
        row = _base(source_file, source_item, image_name, clip_id, frame_id, str(item.get("track_id") or ""))
        row.update(item)
        row.setdefault("raw_label", str(item.get("point_type", "")))
        row.setdefault("normalized_label", str(item.get("point_type", "")))
        data.lux_points.append(row)
    for item in measurement.get("qa_flags", []):
        row = _base(source_file, source_item, image_name, clip_id, frame_id, str(item.get("track_id") or ""))
        flag = str(item.get("flag") or item.get("qa_flag") or "")
        row.update({"qa_flag": flag, "raw_label": flag, "normalized_label": flag, "notes": item.get("notes", "")})
        data.qa_flags.append(row)


def _row_is_invalid(row: dict[str, Any]) -> bool:
    return row.get("validation_status") not in {"", "valid"}


def _validate_references(data: PreparedData) -> None:
    frame_keys = {(str(row.get("clip_id", "")), str(row.get("frame_id", ""))) for row in data.frames}
    track_keys = {
        (str(row.get("clip_id", "")), str(row.get("frame_id", "")), str(row.get("track_id", "")))
        for row in data.tracks
    }
    for table, rows, requires_track in [
        ("tracks", data.tracks, False),
        ("lamp_status", data.lamp_status, True),
        ("affected_regions", data.affected_regions, True),
        ("visibility_labels", data.visibility_labels, True),
        ("attribution_labels", data.attribution_labels, True),
        ("lux_points", data.lux_points, False),
    ]:
        for row in rows:
            frame_key = (str(row.get("clip_id", "")), str(row.get("frame_id", "")))
            if frame_key not in frame_keys:
                _record_issue(data, row, table, "frame_id", row.get("frame_id"), "", "row references a missing frame")
            track_id = str(row.get("track_id", ""))
            if requires_track and not track_id:
                _record_issue(data, row, table, "track_id", "", "", "row requires a target lamp track_id")
            elif requires_track and (*frame_key, track_id) not in track_keys:
                _record_issue(data, row, table, "track_id", track_id, "", "row references a missing track in the same frame")


def _mode_rows(rows: list[dict[str, Any]], validation_mode: str) -> list[dict[str, Any]]:
    if validation_mode == "quarantine":
        return [row for row in rows if not _row_is_invalid(row)]
    return rows


def _write_prepared_dataset(data: PreparedData, out: Path, validation_mode: str) -> None:
    for subdir in ("clips", "frames", "tracks", "annotations", "lux", "validation", "splits"):
        (out / subdir).mkdir(parents=True, exist_ok=True)

    _write_csv(out / "frames" / "frames.csv", _mode_rows(data.frames, validation_mode))
    _write_csv(out / "tracks" / "tracks.csv", _mode_rows(data.tracks, validation_mode))
    _write_csv(out / "annotations" / "lamp_status.csv", _mode_rows(data.lamp_status, validation_mode))
    _write_csv(out / "annotations" / "public_space_regions.csv", _mode_rows(data.public_space_regions, validation_mode))
    _write_csv(out / "annotations" / "affected_regions.csv", _mode_rows(data.affected_regions, validation_mode))
    _write_csv(out / "annotations" / "confounders.csv", _mode_rows(data.confounders, validation_mode))
    _write_csv(out / "annotations" / "visibility_labels.csv", _mode_rows(data.visibility_labels, validation_mode))
    _write_csv(out / "annotations" / "attribution_labels.csv", _mode_rows(data.attribution_labels, validation_mode))
    _write_csv(out / "annotations" / "qa_flags.csv", _mode_rows(data.qa_flags, validation_mode))
    _write_csv(out / "lux" / "lux_points.csv", _mode_rows(data.lux_points, validation_mode))
    _write_csv(out / "validation" / "warnings.csv", [issue.to_row() for issue in data.issues])
    invalid_rows = [row for rows in _all_output_rows(data) for row in rows if _row_is_invalid(row)]
    _write_csv(out / "validation" / "invalid_rows.csv", invalid_rows)

    clip_entries = _write_clip_manifests(data, out, validation_mode)
    report = {
        "validation_mode": validation_mode,
        "frames": len(data.frames),
        "tracks": len(data.tracks),
        "warnings": len(data.issues),
        "invalid_rows": len(invalid_rows),
        "clips": len(clip_entries),
    }
    (out / "validation" / "normalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_type": "rbccps_measurement",
                "version": "pre_dataset_annotations_v1",
                "validation_mode": validation_mode,
                "clips": clip_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _all_output_rows(data: PreparedData) -> list[list[dict[str, Any]]]:
    return [
        data.frames,
        data.tracks,
        data.lamp_status,
        data.public_space_regions,
        data.affected_regions,
        data.confounders,
        data.visibility_labels,
        data.attribution_labels,
        data.qa_flags,
        data.lux_points,
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for name in BASE_COLUMNS:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_clip_manifests(data: PreparedData, out: Path, validation_mode: str) -> list[dict[str, Any]]:
    frames = _mode_rows(data.frames, validation_mode)
    tracks = _mode_rows(data.tracks, validation_mode)
    frames_by_clip: dict[str, list[dict[str, Any]]] = {}
    tracks_by_clip: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        frames_by_clip.setdefault(str(frame["clip_id"]), []).append(frame)
    for track in tracks:
        tracks_by_clip.setdefault(str(track["clip_id"]), []).append(track)

    entries = []
    for clip_id, clip_frames in sorted(frames_by_clip.items()):
        clip_tracks = tracks_by_clip.get(clip_id, [])
        if not clip_frames or not clip_tracks:
            continue
        frame_id_map = {str(frame["frame_id"]): index for index, frame in enumerate(sorted(clip_frames, key=lambda item: str(item["frame_id"])), start=1)}
        frame_payloads = []
        timestamp_by_frame: dict[str, int] = {}
        for source_frame in sorted(clip_frames, key=lambda item: frame_id_map[str(item["frame_id"])]):
            frame_index = frame_id_map[str(source_frame["frame_id"])]
            timestamp_ns = _to_int(source_frame.get("timestamp_ns"), 1_700_000_000_000_000_000 + (frame_index - 1) * 33_333_333)
            timestamp_by_frame[str(source_frame["frame_id"])] = timestamp_ns
            frame_payloads.append(
                {
                    "frame_id": frame_index,
                    "timestamp_ns": timestamp_ns,
                    "image_uri": source_frame.get("image_path") or source_frame.get("image_name") or f"frames/{frame_index:06d}.jpg",
                    "image_format": Path(str(source_frame.get("image_path") or source_frame.get("image_name") or ".jpg")).suffix.lower().lstrip(".") or "unknown",
                    "width": _to_int(source_frame.get("width"), 1),
                    "height": _to_int(source_frame.get("height"), 1),
                    "camera": {
                        "ae_mode": "auto",
                        "hdr_mode": "unknown",
                        "night_mode": True,
                        "metadata_quality": source_frame.get("metadata_quality") or "pseudo",
                    },
                    "pose": {"imu_quality": "missing"},
                }
            )
        track_payloads = []
        for source_track in clip_tracks:
            original_frame_id = str(source_track["frame_id"])
            if original_frame_id not in frame_id_map:
                continue
            try:
                bbox = json.loads(str(source_track.get("bbox_xyxy") or "[]"))
            except json.JSONDecodeError:
                bbox = [0, 0, 1, 1]
            score = _to_float(source_track.get("detector_score"), 0.5)
            track_payloads.append(
                {
                    "frame_id": frame_id_map[original_frame_id],
                    "timestamp_ns": timestamp_by_frame[original_frame_id],
                    "track_id": str(source_track.get("track_id") or "track_001"),
                    "class_name": str(source_track.get("class_name") or "streetlight_lamp_head"),
                    "bbox_xyxy": bbox,
                    "bbox_format": "pixel_xyxy_original_frame",
                    "detector_score": max(0.01, min(1.0, score)),
                    "track_confidence": _to_float(source_track.get("track_confidence"), max(0.01, min(1.0, score))),
                    "track_age": _to_int(source_track.get("track_age"), len(frame_payloads)),
                    "lost_count": _to_int(source_track.get("lost_count"), 0),
                    "source_model": str(source_track.get("source_model") or "annotation_converter"),
                }
            )
        payload = {
            "clip_id": clip_id,
            "device_id": "annotation_converter",
            "calibration_level": 1,
            "policy_id": "rbccps_measurement_policy_v1",
            "video_uri": None,
            "frames": frame_payloads,
            "tracks": track_payloads,
            "optional_calibration": {"photometric": {"field_lux_calibration_id": None}, "map_priors": {}},
        }
        manifest = ClipManifest.from_dict(payload)
        validate_clip_manifest(manifest)
        target = out / "clips" / f"{clip_id}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        entries.append(
            {
                "clip_id": clip_id,
                "device_id": "annotation_converter",
                "calibration_level": 1,
                "manifest": str(target.relative_to(out)).replace("\\", "/"),
                "num_frames": len(frame_payloads),
                "num_tracks": len(track_payloads),
            }
        )
    return entries
