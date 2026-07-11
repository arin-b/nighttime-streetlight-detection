from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from rbccps_od.data_management import seed_sources as seed_utils


from rbccps_od.config.paths import repo_root


ROOT = repo_root()
DATASETS_ROOT = ROOT / "datasets"
V2_ROOT = DATASETS_ROOT / "derived" / "annotation_automation_v2"
CLICK_REVIEW_ROOT = DATASETS_ROOT / "derived" / "annotation_click_review" / "reviews"
CURRENT_NEGATIVE_ROOT = DATASETS_ROOT / "derived" / "annotation_automation" / "reviews"
OUTPUT_ROOT = DATASETS_ROOT / "derived" / "annotation_automation_v3"
REVIEWS_ROOT = OUTPUT_ROOT / "reviews"
REPORTS_ROOT = OUTPUT_ROOT / "reports"


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
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_lookup() -> dict[str, str]:
    rows = load_csv(V2_ROOT / "manifests" / "merged_manifest_v2.csv")
    return {row["image_uid"]: row["assigned_split"] for row in rows}


def positive_review_rows(mode_name: str, records: list[seed_utils.ImageRecord], existing_rows: list[dict], split_map: dict[str, str]) -> list[dict]:
    existing_by_key = {row["key"]: row for row in existing_rows}
    rows: list[dict] = []
    for record in sorted(records, key=lambda item: (item.clip_id, item.frame_id, item.image_uid)):
        if not record.has_annotation:
            continue
        source = existing_by_key.get(record.image_uid)
        review_status = "pending"
        fix_reason = ""
        exclude_reason = ""
        scene_bucket = ""
        updated_boxes_json = ""
        review_timestamp = ""
        review_source = ""

        if source:
            primary = source.get("primary_decision", "").strip()
            secondary = source.get("secondary_reason", "").strip()
            if primary == "keep":
                review_status = "keep"
            elif primary == "fix":
                review_status = "fix_box"
                fix_reason = secondary
            elif primary == "exclude":
                review_status = secondary or "exclude_manual"
                exclude_reason = secondary
            scene_bucket = source.get("scene_bucket", "").strip()
            updated_boxes_json = source.get("updated_boxes_json", "").strip()
            review_timestamp = source.get("review_timestamp", "").strip()
            review_source = "annotation_click_review"

        rows.append(
            {
                "mode": mode_name,
                "key": record.image_uid,
                "image_uid": record.image_uid,
                "dataset_id": record.dataset_id,
                "clip_id": record.clip_id,
                "frame_id": record.frame_id,
                "image_path": str(record.source_image_path),
                "annotation_count": str(record.annotation_count),
                "current_split_v2": split_map.get(record.image_uid, ""),
                "review_status": review_status,
                "fix_reason": fix_reason,
                "exclude_reason": exclude_reason,
                "scene_bucket": scene_bucket,
                "updated_boxes_json": updated_boxes_json,
                "review_timestamp": review_timestamp,
                "review_source": review_source,
            }
        )
    return rows


def negative_review_rows(existing_rows: list[dict]) -> list[dict]:
    review_app_rows = {row["key"]: row for row in load_csv(CLICK_REVIEW_ROOT / "negative_review.csv")}
    existing_v3_rows = {row["key"]: row for row in load_csv(REVIEWS_ROOT / "negative_review_v3.csv")}
    rows: list[dict] = []
    for row in existing_rows:
        source = review_app_rows.get(row["review_candidate_id"])
        existing_v3 = existing_v3_rows.get(row["review_candidate_id"])
        review_status = "pending"
        scene_bucket = ""
        updated_boxes_json = ""
        review_timestamp = ""
        review_source = ""

        if source:
            review_status = source.get("primary_decision", "pending").strip() or "pending"
            scene_bucket = source.get("scene_bucket", "").strip()
            updated_boxes_json = source.get("updated_boxes_json", "").strip()
            review_timestamp = source.get("review_timestamp", "").strip()
            review_source = "annotation_click_review"
        else:
            label = row.get("review_label", "").strip()
            if label == "clean_negative":
                review_status = "clean_negative"
            elif label == "missed_positive":
                review_status = "promote_to_positive_review"
            elif label == "ambiguous":
                review_status = "exclude_ambiguous_visibility"
            review_source = "hard_negative_review_manifest"
            if existing_v3:
                scene_bucket = existing_v3.get("scene_bucket", "").strip()
                updated_boxes_json = existing_v3.get("updated_boxes_json", "").strip()
                review_timestamp = existing_v3.get("review_timestamp", "").strip()
                if existing_v3.get("review_status", "").strip():
                    review_status = existing_v3["review_status"].strip()
                if existing_v3.get("review_source", "").strip():
                    review_source = existing_v3["review_source"].strip()

        rows.append(
            {
                "mode": "negative_review",
                "key": row["review_candidate_id"],
                "review_candidate_id": row["review_candidate_id"],
                "source_pool": row["source_pool"],
                "dataset_id": row["dataset_id"],
                "clip_id": row["clip_id"],
                "frame_id": row["frame_id"],
                "image_path": row["image_path"],
                "priority": row.get("priority", ""),
                "suggested_negative_subtype": row.get("suggested_negative_subtype", ""),
                "review_status": review_status,
                "scene_bucket": scene_bucket,
                "updated_boxes_json": updated_boxes_json,
                "review_timestamp": review_timestamp,
                "review_source": review_source,
                "notes": row.get("notes", ""),
            }
        )
    return rows


def secondary_exports(jobin_rows: list[dict], arindam_rows: list[dict], negative_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    scene_rows: list[dict] = []
    promoted_rows: list[dict] = []
    for row in jobin_rows + arindam_rows + negative_rows:
        if row.get("scene_bucket"):
            scene_rows.append(
                {
                    "mode": row["mode"],
                    "key": row["key"],
                    "dataset_id": row["dataset_id"],
                    "clip_id": row["clip_id"],
                    "frame_id": row["frame_id"],
                    "scene_bucket": row["scene_bucket"],
                    "review_status": row["review_status"],
                    "review_timestamp": row["review_timestamp"],
                }
            )
        if row["mode"] == "negative_review" and row["review_status"] == "promote_to_positive_review":
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
    return scene_rows, promoted_rows


def main() -> None:
    ensure_dir(REVIEWS_ROOT)
    ensure_dir(REPORTS_ROOT)
    split_map = split_lookup()

    jobin_records, _ = seed_utils.load_jobin_source()
    arindam_records, _ = seed_utils.load_arindam_source()
    current_jobin_rows = load_csv(CLICK_REVIEW_ROOT / "jobin_positive.csv")
    current_arindam_rows = load_csv(CLICK_REVIEW_ROOT / "arindam_positive.csv")
    current_negative_rows = load_csv(CURRENT_NEGATIVE_ROOT / "hard_negative_review_manifest.csv")

    jobin_rows = positive_review_rows("jobin_positive", jobin_records, current_jobin_rows, split_map)
    arindam_rows = positive_review_rows("arindam_positive", arindam_records, current_arindam_rows, split_map)
    negative_rows = negative_review_rows(current_negative_rows)
    scene_rows, promoted_rows = secondary_exports(jobin_rows, arindam_rows, negative_rows)

    write_csv(
        REVIEWS_ROOT / "jobin_positive_review_v3.csv",
        jobin_rows,
        [
            "mode",
            "key",
            "image_uid",
            "dataset_id",
            "clip_id",
            "frame_id",
            "image_path",
            "annotation_count",
            "current_split_v2",
            "review_status",
            "fix_reason",
            "exclude_reason",
            "scene_bucket",
            "updated_boxes_json",
            "review_timestamp",
            "review_source",
        ],
    )
    write_csv(
        REVIEWS_ROOT / "arindam_positive_review_v3.csv",
        arindam_rows,
        [
            "mode",
            "key",
            "image_uid",
            "dataset_id",
            "clip_id",
            "frame_id",
            "image_path",
            "annotation_count",
            "current_split_v2",
            "review_status",
            "fix_reason",
            "exclude_reason",
            "scene_bucket",
            "updated_boxes_json",
            "review_timestamp",
            "review_source",
        ],
    )
    write_csv(
        REVIEWS_ROOT / "negative_review_v3.csv",
        negative_rows,
        [
            "mode",
            "key",
            "review_candidate_id",
            "source_pool",
            "dataset_id",
            "clip_id",
            "frame_id",
            "image_path",
            "priority",
            "suggested_negative_subtype",
            "review_status",
            "scene_bucket",
            "updated_boxes_json",
            "review_timestamp",
            "review_source",
            "notes",
        ],
    )
    write_csv(
        REVIEWS_ROOT / "scene_bucket_manifest_v3.csv",
        scene_rows,
        ["mode", "key", "dataset_id", "clip_id", "frame_id", "scene_bucket", "review_status", "review_timestamp"],
    )
    write_csv(
        REVIEWS_ROOT / "promoted_positive_queue_v3.csv",
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

    summary = {
        "jobin_positive_review_v3": dict(Counter(row["review_status"] for row in jobin_rows)),
        "arindam_positive_review_v3": dict(Counter(row["review_status"] for row in arindam_rows)),
        "negative_review_v3": dict(Counter(row["review_status"] for row in negative_rows)),
        "promoted_positive_queue_v3": len(promoted_rows),
        "scene_bucket_rows_v3": len(scene_rows),
    }
    write_json(REPORTS_ROOT / "review_progress_v3.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
