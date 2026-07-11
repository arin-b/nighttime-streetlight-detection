from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATOR_SCHEMA_VERSION = "measurement_annotator_v1"
LEGACY_STREETLIGHT_CLASS = "streetlight"
LAMP_HEAD_CLASS = "streetlight_lamp_head"
POLE_CLASS = "streetlight_pole"
LAMP_BOX_CLASSES = [LAMP_HEAD_CLASS, POLE_CLASS]

SURFACE_TYPES = [
    "building_facade",
    "shopfront",
    "window",
    "sign_lightbox",
    "reflective_glass",
    "wet_road_reflection",
    "wall_compound_surface",
    "vehicle_headlight_region",
    "unknown_bright_source",
]

PUBLIC_SPACE_TYPES = [
    "road",
    "footpath_sidewalk",
    "crossing",
    "curb",
    "median",
    "verge",
    "vegetation",
    "vehicle",
    "building_frontage",
    "shopfront",
    "window",
    "sign_billboard",
    "traffic_signal",
    "sky",
    "wet_reflection_like_road",
    "occluder",
    "unknown",
]

LAMP_STATUS_CLASSES = ["on", "dim", "off", "flicker", "occluded", "saturated", "unknown"]
VISIBILITY_CLASSES = ["good", "adequate", "marginal", "poor", "dark", "unknown"]
ATTRIBUTION_CLASSES = ["certain", "mixed", "uncertain", "impossible_due_to_confounding"]
LUX_POINT_TYPES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]


def normalize_lamp_class(value: str | None) -> str:
    if value in {None, "", LEGACY_STREETLIGHT_CLASS}:
        return LAMP_HEAD_CLASS
    return str(value)


def sanitize_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "item"


@dataclass
class TrackRecord:
    track_id: str
    bbox_xyxy: list[float]
    class_name: str = LAMP_HEAD_CLASS
    detector_score: float | None = None
    track_confidence: float | None = None
    track_age: int | None = None
    lost_count: int | None = None
    source_model: str | None = None


@dataclass
class AnnotationItem:
    key: str
    image_id: str
    image_path: str
    width: int
    height: int
    dataset_id: str = "local"
    route_id: str = ""
    clip_id: str = ""
    frame_id: str = ""
    timestamp_ns: int | None = None
    split: str = "unassigned"
    source_pool: str = "raw_frames"
    tracks: list[TrackRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tracks"] = [asdict(track) for track in self.tracks]
        return payload


@dataclass
class WorkspaceManifest:
    schema_version: str
    workspace_id: str
    created_at: str
    source_type: str
    items: list[AnnotationItem]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "source_type": self.source_type,
            "notes": self.notes,
            "items": [item.to_dict() for item in self.items],
        }


def default_review(item: AnnotationItem) -> dict[str, Any]:
    boxes = []
    for index, track in enumerate(item.tracks, start=1):
        boxes.append(
            {
                "box_id": f"box_{index:03d}",
                "class_name": normalize_lamp_class(track.class_name),
                "bbox_xyxy": track.bbox_xyxy,
                "track_id": track.track_id,
                "parent_pole_box_id": "",
                "status": "candidate",
                "source": "detector_track",
                "notes": "",
            }
        )
    return {
        "schema_version": ANNOTATOR_SCHEMA_VERSION,
        "item_key": item.key,
        "review_status": "unreviewed",
        "boxes": boxes,
        "confounder_boxes": [],
        "polygons": [],
        "measurement": {
            "lamp_status": [],
            "public_space_regions": [],
            "affected_regions": [],
            "visibility_labels": [],
            "attribution_labels": [],
            "lux_points": [],
            "qa_flags": [],
        },
        "updated_at": "",
    }


def workspace_for_frames(frames: Path) -> Path:
    return frames.resolve().parent / "measurement_annotation_workspace"
