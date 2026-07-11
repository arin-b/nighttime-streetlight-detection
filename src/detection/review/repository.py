from __future__ import annotations

import csv
import json
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_od.data_management import seed_sources as seed_utils


from rbccps_od.config.paths import repo_root


ROOT = repo_root()
DATASETS_ROOT = ROOT / "datasets"
CURRENT_REVIEWS_ROOT = DATASETS_ROOT / "derived" / "annotation_automation" / "reviews"
CURRENT_V2_ROOT = DATASETS_ROOT / "derived" / "annotation_automation_v2"
OUTPUT_ROOT = DATASETS_ROOT / "derived" / "annotation_click_review"
APP_DATA_ROOT = OUTPUT_ROOT / "app_data"
REVIEWS_ROOT = OUTPUT_ROOT / "reviews"

MODE_ORDER = ["jobin_positive", "arindam_positive", "negative_review"]

MODE_BASELINE_COMPLETED = {
    "jobin_positive": 87,
}

MODE_LABELS = {
    "jobin_positive": "Jobin Positive Review",
    "arindam_positive": "Arindam Positive Review",
    "negative_review": "Hard-Negative Review",
}

POSITIVE_DECISIONS = [
    {"id": "keep", "label": "Keep"},
    {"id": "fix", "label": "Fix"},
    {"id": "exclude", "label": "Exclude"},
]

FIX_REASONS = [
    {"id": "fix_full_visible_luminaire_extent", "label": "Fix Full Luminaire Extent"},
    {"id": "fix_loose_dark_region", "label": "Fix Loose Dark Region"},
    {"id": "fix_pole_tree_merge", "label": "Fix Pole/Tree Merge"},
    {"id": "fix_off_center", "label": "Fix Off Center"},
    {"id": "fix_box_too_small", "label": "Fix Box Too Small"},
    {"id": "fix_box_too_large", "label": "Fix Box Too Large"},
]

EXCLUDE_REASONS = [
    {"id": "exclude_occluded", "label": "Exclude Occluded"},
    {"id": "exclude_truncated", "label": "Exclude Truncated"},
    {"id": "exclude_glare_blob", "label": "Exclude Glare Blob"},
    {"id": "exclude_ambiguous_source", "label": "Exclude Ambiguous Source"},
    {"id": "exclude_pole_tree_dominated", "label": "Exclude Pole/Tree Dominated"},
    {"id": "exclude_not_full_visible_luminaire", "label": "Exclude Not Full Visible Luminaire"},
]

SCENE_BUCKETS = [
    {"id": "quiet_residential_lane", "label": "Quiet Residential Lane"},
    {"id": "busy_arterial_road", "label": "Busy Arterial Road"},
    {"id": "heavy_glare_traffic", "label": "Heavy Glare/Traffic"},
]

NEGATIVE_DECISIONS = [
    {"id": "clean_negative", "label": "Clean Negative"},
    {"id": "promote_to_positive_review", "label": "Promote to Positive Review"},
    {"id": "exclude_ambiguous_visibility", "label": "Exclude Ambiguous Visibility"},
]

FRAME_NUM_RE = re.compile(r"(\d+)")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def frame_sort_key(value: str) -> tuple[int, str]:
    match = FRAME_NUM_RE.search(value)
    if not match:
        return (0, value)
    return (int(match.group(1)), value)


def output_fieldnames_for_mode(mode: str) -> list[str]:
    if mode == "negative_review":
        return [
            "key",
            "review_candidate_id",
            "source_pool",
            "dataset_id",
            "clip_id",
            "frame_id",
            "image_path",
            "primary_decision",
            "secondary_reason",
            "scene_bucket",
            "updated_boxes_json",
            "review_timestamp",
        ]
    return [
        "key",
        "image_uid",
        "dataset_id",
        "clip_id",
        "frame_id",
        "image_path",
        "annotation_count",
        "current_split_v2",
        "primary_decision",
        "secondary_reason",
        "scene_bucket",
        "updated_boxes_json",
        "review_timestamp",
    ]


def mode_csv_path(mode: str) -> Path:
    return REVIEWS_ROOT / f"{mode}.csv"


def mode_json_path(mode: str) -> Path:
    return APP_DATA_ROOT / f"{mode}.json"


def mode_subset_keys_path(mode: str) -> Path:
    return REVIEWS_ROOT / f"{mode}_subset_keys.txt"


def signoff_path() -> Path:
    return REVIEWS_ROOT / "mode_signoff.json"


@dataclass
class ReviewItem:
    key: str
    review_id: str
    dataset_id: str
    clip_id: str
    frame_id: str
    image_path: str
    width: int
    height: int
    boxes: list[list[float]]
    annotation_count: int
    current_split_v2: str
    source_pool: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "review_id": self.review_id,
            "dataset_id": self.dataset_id,
            "clip_id": self.clip_id,
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "boxes": self.boxes,
            "annotation_count": self.annotation_count,
            "current_split_v2": self.current_split_v2,
            "source_pool": self.source_pool,
        }


class ReviewRepository:
    def __init__(self) -> None:
        ensure_dir(APP_DATA_ROOT)
        ensure_dir(REVIEWS_ROOT)
        self.lock = threading.Lock()
        self.mode_items = self._build_items()
        self.reviews_by_mode = self._load_reviews()
        self.signoffs = self._load_signoffs()
        self._write_app_data()
        for mode in MODE_ORDER:
            self._rewrite_mode_csv(mode)
        self._rewrite_secondary_exports()

    def _build_items(self) -> dict[str, list[ReviewItem]]:
        split_lookup: dict[str, str] = {}
        for row in load_csv(CURRENT_V2_ROOT / "manifests" / "merged_manifest_v2.csv"):
            split_lookup[row["image_uid"]] = row["assigned_split"]

        jobin_records, jobin_annotations = seed_utils.load_jobin_source()
        arindam_records, arindam_annotations = seed_utils.load_arindam_source()

        items_by_mode: dict[str, list[ReviewItem]] = {
            "jobin_positive": [],
            "arindam_positive": [],
            "negative_review": [],
        }

        for dataset_name, records, annotations, mode in (
            ("jobin", jobin_records, jobin_annotations, "jobin_positive"),
            ("arindam", arindam_records, arindam_annotations, "arindam_positive"),
        ):
            boxes_by_uid: dict[str, list[list[float]]] = defaultdict(list)
            for ann in annotations:
                boxes_by_uid[ann["image_uid"]].append(seed_utils.coerce_bbox(ann["bbox"]))

            positive_records = [
                record
                for record in records
                if record.dataset_id == dataset_name and record.has_annotation and boxes_by_uid.get(record.image_uid)
            ]
            positive_records.sort(key=lambda item: (item.clip_id, frame_sort_key(item.frame_id), item.image_uid))
            for index, record in enumerate(positive_records, start=1):
                items_by_mode[mode].append(
                    ReviewItem(
                        key=record.image_uid,
                        review_id=f"{dataset_name}_pos_{index:05d}",
                        dataset_id=record.dataset_id,
                        clip_id=record.clip_id,
                        frame_id=record.frame_id,
                        image_path=str(record.source_image_path),
                        width=record.width,
                        height=record.height,
                        boxes=boxes_by_uid[record.image_uid],
                        annotation_count=record.annotation_count,
                        current_split_v2=split_lookup.get(record.image_uid, ""),
                    )
                )

        negative_rows = load_csv(CURRENT_REVIEWS_ROOT / "hard_negative_review_manifest.csv")
        negative_rows.sort(key=lambda row: (row["source_pool"], row["clip_id"], frame_sort_key(row["frame_id"]), row["review_candidate_id"]))
        for row in negative_rows:
            items_by_mode["negative_review"].append(
                ReviewItem(
                    key=row["review_candidate_id"],
                    review_id=row["review_candidate_id"],
                    dataset_id=row["dataset_id"],
                    clip_id=row["clip_id"],
                    frame_id=row["frame_id"],
                    image_path=row["image_path"],
                    width=0,
                    height=0,
                    boxes=[],
                    annotation_count=0,
                    current_split_v2="",
                    source_pool=row["source_pool"],
                )
            )

        return items_by_mode

    def _load_reviews(self) -> dict[str, dict[str, dict[str, str]]]:
        reviews: dict[str, dict[str, dict[str, str]]] = {mode: {} for mode in MODE_ORDER}
        for mode in MODE_ORDER:
            for row in load_csv(mode_csv_path(mode)):
                reviews[mode][row["key"]] = row
        return reviews

    def _load_signoffs(self) -> dict[str, bool]:
        payload = read_json(signoff_path(), {})
        return {mode: bool(payload.get(mode, False)) for mode in MODE_ORDER}

    def _load_subset_keys(self, mode: str) -> set[str]:
        path = mode_subset_keys_path(mode)
        if not path.exists():
            return set()
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def _write_signoffs(self) -> None:
        write_json(signoff_path(), self.signoffs)

    def _write_app_data(self) -> None:
        for mode in MODE_ORDER:
            write_json(mode_json_path(mode), [item.to_dict() for item in self.mode_items[mode]])

    def _rewrite_mode_csv(self, mode: str) -> None:
        fieldnames = output_fieldnames_for_mode(mode)
        order_lookup = {item.key: index for index, item in enumerate(self.mode_items[mode])}
        rows = sorted(self.reviews_by_mode[mode].values(), key=lambda row: order_lookup.get(row["key"], 10**9))
        write_csv(mode_csv_path(mode), rows, fieldnames)

    def _rewrite_secondary_exports(self) -> None:
        promoted_rows: list[dict[str, str]] = []
        excluded_rows: list[dict[str, str]] = []
        scene_rows: list[dict[str, str]] = []

        for mode in MODE_ORDER:
            for row in self.reviews_by_mode[mode].values():
                if row.get("scene_bucket"):
                    scene_rows.append(
                        {
                            "mode": mode,
                            "key": row["key"],
                            "dataset_id": row["dataset_id"],
                            "clip_id": row["clip_id"],
                            "frame_id": row["frame_id"],
                            "scene_bucket": row["scene_bucket"],
                            "primary_decision": row["primary_decision"],
                            "secondary_reason": row.get("secondary_reason", ""),
                            "review_timestamp": row["review_timestamp"],
                        }
                    )
                if mode == "negative_review" and row["primary_decision"] == "promote_to_positive_review":
                    promoted_rows.append(
                        {
                            "source_review_key": row["key"],
                            "review_candidate_id": row["review_candidate_id"],
                            "source_pool": row["source_pool"],
                            "dataset_id": row["dataset_id"],
                            "clip_id": row["clip_id"],
                            "frame_id": row["frame_id"],
                            "image_path": row["image_path"],
                            "scene_bucket": row["scene_bucket"],
                            "review_timestamp": row["review_timestamp"],
                        }
                    )
                if row["primary_decision"] == "exclude" or row["primary_decision"] == "exclude_ambiguous_visibility":
                    excluded_rows.append(
                        {
                            "mode": mode,
                            "key": row["key"],
                            "dataset_id": row["dataset_id"],
                            "clip_id": row["clip_id"],
                            "frame_id": row["frame_id"],
                            "image_path": row["image_path"],
                            "primary_decision": row["primary_decision"],
                            "secondary_reason": row.get("secondary_reason", ""),
                            "scene_bucket": row.get("scene_bucket", ""),
                            "review_timestamp": row["review_timestamp"],
                        }
                    )

        write_csv(
            REVIEWS_ROOT / "promoted_positive_queue.csv",
            promoted_rows,
            [
                "source_review_key",
                "review_candidate_id",
                "source_pool",
                "dataset_id",
                "clip_id",
                "frame_id",
                "image_path",
                "scene_bucket",
                "review_timestamp",
            ],
        )
        write_csv(
            REVIEWS_ROOT / "excluded_items.csv",
            excluded_rows,
            [
                "mode",
                "key",
                "dataset_id",
                "clip_id",
                "frame_id",
                "image_path",
                "primary_decision",
                "secondary_reason",
                "scene_bucket",
                "review_timestamp",
            ],
        )
        write_csv(
            REVIEWS_ROOT / "scene_bucket_audit.csv",
            scene_rows,
            [
                "mode",
                "key",
                "dataset_id",
                "clip_id",
                "frame_id",
                "scene_bucket",
                "primary_decision",
                "secondary_reason",
                "review_timestamp",
            ],
        )

    def mode_status(self, mode: str) -> dict[str, Any]:
        total = len(self.mode_items[mode])
        baseline_completed = min(MODE_BASELINE_COMPLETED.get(mode, 0), total)
        active_keys = self._active_ordered_keys(mode)
        active_items = [item for item in self.mode_items[mode] if item.key in set(active_keys)]
        reviewed = sum(
            1
            for index, item in enumerate(self.mode_items[mode], start=1)
            if index <= baseline_completed or item.key in self.reviews_by_mode[mode]
        )
        active_reviewed = sum(1 for item in active_items if item.key in self.reviews_by_mode[mode])
        active_pending = sum(1 for item in active_items if item.key not in self.reviews_by_mode[mode])
        subset_enabled = bool(self._load_subset_keys(mode))
        stage_complete = (reviewed >= total) or (subset_enabled and active_pending == 0)
        previous_modes = MODE_ORDER[: MODE_ORDER.index(mode)]
        unlocked = all(self.signoffs.get(previous_mode, False) for previous_mode in previous_modes)
        return {
            "mode": mode,
            "label": MODE_LABELS[mode],
            "total": total,
            "reviewed": reviewed,
            "remaining": total - reviewed,
            "complete": reviewed >= total,
            "stage_complete": stage_complete,
            "signed_off": self.signoffs.get(mode, False),
            "unlocked": unlocked,
            "baseline_completed": baseline_completed,
            "active_total": len(active_keys),
            "active_reviewed": active_reviewed,
            "active_pending": active_pending,
            "subset_enabled": subset_enabled,
        }

    def _ordered_keys(self, mode: str) -> list[str]:
        return [item.key for item in self.mode_items[mode]]

    def _active_ordered_keys(self, mode: str) -> list[str]:
        ordered_keys = self._ordered_keys(mode)
        baseline_completed = MODE_BASELINE_COMPLETED.get(mode, 0)
        active_keys = ordered_keys[baseline_completed:]
        subset_keys = self._load_subset_keys(mode)
        if not subset_keys:
            return active_keys
        reviewed_keys = set(self.reviews_by_mode[mode])
        filtered: list[str] = []
        for key in active_keys:
            if key in reviewed_keys or key in subset_keys:
                filtered.append(key)
        return filtered

    def bootstrap(self) -> dict[str, Any]:
        statuses = [self.mode_status(mode) for mode in MODE_ORDER]
        current_mode = next((status["mode"] for status in statuses if status["unlocked"] and not status["signed_off"]), MODE_ORDER[-1])
        return {
            "modes": statuses,
            "current_mode": current_mode,
            "positive_decisions": POSITIVE_DECISIONS,
            "fix_reasons": FIX_REASONS,
            "exclude_reasons": EXCLUDE_REASONS,
            "scene_buckets": SCENE_BUCKETS,
            "negative_decisions": NEGATIVE_DECISIONS,
        }

    def get_item(self, mode: str, key: str | None = None) -> dict[str, Any] | None:
        if mode not in self.mode_items:
            return None
        item_lookup = {item.key: item for item in self.mode_items[mode]}
        ordered_keys = self._ordered_keys(mode)
        active_keys = self._active_ordered_keys(mode)
        if key:
            if key not in active_keys:
                return None
            item = item_lookup.get(key)
            if not item:
                return None
        else:
            item = next((item_lookup[item_key] for item_key in active_keys if item_key not in self.reviews_by_mode[mode]), None)
            if not item:
                return None
        current_index = ordered_keys.index(item.key)
        current_active_index = active_keys.index(item.key)
        prev_key = active_keys[current_active_index - 1] if current_active_index > 0 else None
        next_key = active_keys[current_active_index + 1] if current_active_index + 1 < len(active_keys) else None
        payload = item.to_dict()
        payload["existing_review"] = self.reviews_by_mode[mode].get(item.key)
        payload["mode_status"] = self.mode_status(mode)
        payload["position"] = current_index + 1
        payload["active_total"] = len(active_keys)
        payload["active_position"] = current_active_index + 1
        payload["prev_key"] = prev_key
        payload["next_key"] = next_key
        return payload

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload["mode"]
        if mode not in MODE_ORDER:
            raise ValueError("Unknown mode.")
        item_lookup = {item.key: item for item in self.mode_items[mode]}
        key = payload["key"]
        item = item_lookup.get(key)
        if not item:
            raise ValueError("Unknown review item.")

        primary_decision = payload.get("primary_decision", "").strip()
        secondary_reason = payload.get("secondary_reason", "").strip()
        scene_bucket = payload.get("scene_bucket", "").strip()
        updated_boxes = payload.get("updated_boxes", [])

        if mode == "negative_review":
            allowed_primary = {item["id"] for item in NEGATIVE_DECISIONS}
            if primary_decision not in allowed_primary:
                raise ValueError("Invalid negative review decision.")
            if not scene_bucket:
                raise ValueError("Scene bucket is required.")
        else:
            allowed_primary = {item["id"] for item in POSITIVE_DECISIONS}
            if primary_decision not in allowed_primary:
                raise ValueError("Invalid positive review decision.")
            if primary_decision == "fix":
                allowed_secondary = {item["id"] for item in FIX_REASONS}
                if secondary_reason not in allowed_secondary:
                    raise ValueError("Fix reason is required.")
                if not isinstance(updated_boxes, list) or not updated_boxes:
                    raise ValueError("At least one replacement box is required for fixes.")
            elif primary_decision == "exclude":
                allowed_secondary = {item["id"] for item in EXCLUDE_REASONS}
                if secondary_reason not in allowed_secondary:
                    raise ValueError("Exclude reason is required.")
            if not scene_bucket:
                raise ValueError("Scene bucket is required.")

        row: dict[str, str] = {
            "key": item.key,
            "dataset_id": item.dataset_id,
            "clip_id": item.clip_id,
            "frame_id": item.frame_id,
            "image_path": item.image_path,
            "primary_decision": primary_decision,
            "secondary_reason": secondary_reason,
            "scene_bucket": scene_bucket,
            "updated_boxes_json": json.dumps(updated_boxes if primary_decision == "fix" else item.boxes),
            "review_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if mode == "negative_review":
            row.update(
                {
                    "review_candidate_id": item.key,
                    "source_pool": item.source_pool,
                }
            )
        else:
            row.update(
                {
                    "image_uid": item.key,
                    "annotation_count": str(item.annotation_count),
                    "current_split_v2": item.current_split_v2,
                }
            )

        with self.lock:
            self.reviews_by_mode[mode][item.key] = row
            self._rewrite_mode_csv(mode)
            self._rewrite_secondary_exports()
        return self.mode_status(mode)

    def signoff_mode(self, mode: str) -> dict[str, Any]:
        status = self.mode_status(mode)
        if not status["unlocked"]:
            raise ValueError("Mode is locked until prior signoff is complete.")
        if not status["stage_complete"]:
            raise ValueError("Mode cannot be signed off until the active review queue is complete.")
        with self.lock:
            self.signoffs[mode] = True
            self._write_signoffs()
        return self.bootstrap()


