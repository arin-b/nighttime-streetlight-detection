from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from rbccps_od.data_management import seed_sources as seed_utils


from rbccps_od.config.paths import repo_root


ROOT = repo_root()
DATASETS_ROOT = ROOT / "datasets"
SOURCE_V2_ROOT = DATASETS_ROOT / "derived" / "annotation_automation_v2"
OUTPUT_ROOT = DATASETS_ROOT / "derived" / "annotation_automation_v3"
REVIEWS_ROOT = OUTPUT_ROOT / "reviews"

EXCLUDE_STATUSES = {
    "exclude_occluded",
    "exclude_truncated",
    "exclude_glare_blob",
    "exclude_box_too_loose",
    "exclude_manual",
    "exclude_not_full_visible_luminaire",
    "exclude_ambiguous_source",
    "exclude_pole_tree_dominated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local-only annotation_automation_v3 corpus from review manifests.")
    parser.add_argument("--allow-unreviewed-positives", action="store_true", help="Allow unreviewed positives to pass through as original seed labels.")
    parser.add_argument("--allow-missing-scene-buckets", action="store_true", help="Allow retained items without a scene bucket.")
    parser.add_argument("--valid-negative-target", type=int, default=15, help="Minimum validation clean negatives when assigning newly reviewed negatives.")
    parser.add_argument("--test-negative-target", type=int, default=15, help="Minimum test clean negatives when assigning newly reviewed negatives.")
    return parser.parse_args()


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


def parse_boxes(text: str) -> list[list[float]]:
    if not text.strip():
        return []
    payload = json.loads(text)
    return [[float(value) for value in box] for box in payload]


def output_dataset_yaml(dataset_root: Path) -> None:
    dataset_yaml = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            "",
            "names:",
            "  0: streetlight",
            "",
        ]
    )
    (dataset_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")


def yolo_line(bbox: list[float], width: int, height: int) -> str:
    x, y, w, h = bbox
    x_c = (x + (w / 2.0)) / width
    y_c = (y + (h / 2.0)) / height
    return f"0 {x_c:.6f} {y_c:.6f} {w / width:.6f} {h / height:.6f}"


def copy_image(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)


def split_lookup() -> dict[str, str]:
    rows = load_csv(SOURCE_V2_ROOT / "manifests" / "merged_manifest_v2.csv")
    return {row["image_uid"]: row["assigned_split"] for row in rows}


def load_source_records() -> tuple[list[seed_utils.ImageRecord], dict[str, list[list[float]]]]:
    jobin_records, jobin_annotations = seed_utils.load_jobin_source()
    arindam_records, arindam_annotations = seed_utils.load_arindam_source()
    records = [record for record in (jobin_records + arindam_records) if record.has_annotation]
    boxes_by_uid: dict[str, list[list[float]]] = defaultdict(list)
    for ann in jobin_annotations + arindam_annotations:
        boxes_by_uid[ann["image_uid"]].append(seed_utils.coerce_bbox(ann["bbox"]))
    return records, boxes_by_uid


def review_lookup() -> dict[str, dict]:
    rows = load_csv(REVIEWS_ROOT / "jobin_positive_review_v3.csv") + load_csv(REVIEWS_ROOT / "arindam_positive_review_v3.csv")
    return {row["image_uid"]: row for row in rows}


def negative_review_lookup() -> list[dict]:
    return load_csv(REVIEWS_ROOT / "negative_review_v3.csv")


def existing_negative_admissions() -> dict[str, str]:
    rows = load_csv(SOURCE_V2_ROOT / "integrated_reviews" / "reviewed_negative_admissions_v2.csv")
    return {row["review_candidate_id"]: row["assigned_split_v2"] for row in rows}


def scene_bucket_lookup() -> dict[str, str]:
    rows = load_csv(REVIEWS_ROOT / "scene_bucket_manifest_v3.csv")
    return {row["key"]: row["scene_bucket"] for row in rows}


def assign_new_negative_split(valid_count: int, test_count: int, valid_target: int, test_target: int) -> str:
    if valid_count < valid_target:
        return "valid"
    if test_count < test_target:
        return "test"
    return "train"


def build_positive_rows(
    records: list[seed_utils.ImageRecord],
    original_boxes: dict[str, list[list[float]]],
    review_rows: dict[str, dict],
    split_map: dict[str, str],
    allow_unreviewed: bool,
    allow_missing_scene: bool,
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    resolved_rows: list[dict] = []
    manifest_rows: list[dict] = []
    validation_rows: list[dict] = []
    blockers: list[str] = []

    for record in sorted(records, key=lambda item: (split_map.get(item.image_uid, ""), item.dataset_id, item.clip_id, item.frame_id)):
        review = review_rows.get(record.image_uid)
        split = split_map.get(record.image_uid, "")
        if not split:
            blockers.append(f"Missing split assignment for {record.image_uid}")
            continue

        if review:
            review_status = review["review_status"]
        else:
            review_status = "pending"

        scene_bucket = review.get("scene_bucket", "").strip() if review else ""
        if review_status == "pending":
            if not allow_unreviewed:
                blockers.append(f"Pending positive review: {record.image_uid}")
                continue
            boxes = original_boxes[record.image_uid]
            corpus_role = "seed_positive_unreviewed"
        elif review_status == "keep":
            assert review is not None
            boxes = parse_boxes(review.get("updated_boxes_json", "")) or original_boxes[record.image_uid]
            corpus_role = "seed_positive_reviewed_keep"
        elif review_status == "fix_box":
            assert review is not None
            boxes = parse_boxes(review.get("updated_boxes_json", ""))
            if not boxes:
                blockers.append(f"Fix row missing updated boxes: {record.image_uid}")
                continue
            corpus_role = "seed_positive_reviewed_fix"
        elif review_status in EXCLUDE_STATUSES:
            continue
        else:
            blockers.append(f"Unknown positive review status {review_status} for {record.image_uid}")
            continue

        if not allow_missing_scene and review_status in {"keep", "fix_box"} and not scene_bucket:
            blockers.append(f"Missing scene bucket for reviewed retained positive: {record.image_uid}")
            continue

        resolved_rows.append(
            {
                "image_uid": record.image_uid,
                "dataset_id": record.dataset_id,
                "clip_id": record.clip_id,
                "frame_id": record.frame_id,
                "assigned_split": split,
                "corpus_role": corpus_role,
                "scene_bucket": scene_bucket,
                "image_path": str(record.source_image_path),
                "output_file_name": record.output_file_name,
                "width": record.width,
                "height": record.height,
                "boxes": boxes,
                "review_status": review_status,
            }
        )

        manifest_rows.append(
            {
                "image_uid": record.image_uid,
                "dataset_id": record.dataset_id,
                "clip_id": record.clip_id,
                "frame_id": record.frame_id,
                "assigned_split": split,
                "corpus_role": corpus_role,
                "scene_bucket": scene_bucket,
                "image_path": str(record.source_image_path),
                "output_file_name": record.output_file_name,
                "annotation_count": str(len(boxes)),
                "review_status": review_status,
            }
        )
        if split == "valid":
            validation_rows.append(
                {
                    "image_uid": record.image_uid,
                    "dataset_id": record.dataset_id,
                    "clip_id": record.clip_id,
                    "frame_id": record.frame_id,
                    "validation_role": "positive",
                    "condition_group": scene_bucket or "scene_bucket_missing",
                    "image_path": str(record.source_image_path),
                }
            )

    return resolved_rows, manifest_rows, validation_rows, blockers


def build_negative_rows(
    negative_reviews: list[dict],
    existing_admissions: dict[str, str],
    allow_missing_scene: bool,
    valid_target: int,
    test_target: int,
) -> tuple[list[dict], list[dict], list[str]]:
    clean_rows: list[dict] = []
    validation_rows: list[dict] = []
    blockers: list[str] = []

    valid_count = sum(1 for split in existing_admissions.values() if split == "valid")
    test_count = sum(1 for split in existing_admissions.values() if split == "test")

    for row in negative_reviews:
        status = row["review_status"]
        if status == "pending":
            continue
        if status == "promote_to_positive_review":
            continue
        if status != "clean_negative":
            continue

        scene_bucket = row.get("scene_bucket", "").strip()
        if not allow_missing_scene and not scene_bucket:
            blockers.append(f"Missing scene bucket for retained negative: {row['review_candidate_id']}")
            continue

        if row["review_candidate_id"] in existing_admissions:
            split = existing_admissions[row["review_candidate_id"]]
        elif row["source_pool"] == "arindam_unannotated_seed":
            split = "train"
        else:
            split = assign_new_negative_split(valid_count, test_count, valid_target, test_target)
            if split == "valid":
                valid_count += 1
            elif split == "test":
                test_count += 1

        clean_rows.append(
            {
                "review_candidate_id": row["review_candidate_id"],
                "dataset_id": row["dataset_id"],
                "clip_id": row["clip_id"],
                "frame_id": row["frame_id"],
                "assigned_split": split,
                "corpus_role": "reviewed_clean_negative",
                "scene_bucket": scene_bucket,
                "source_pool": row["source_pool"],
                "image_path": row["image_path"],
                "output_file_name": f"neg__{row['review_candidate_id']}__{Path(row['image_path']).stem}{Path(row['image_path']).suffix.lower()}",
                "review_status": status,
            }
        )

        if split == "valid":
            validation_rows.append(
                {
                    "image_uid": row["review_candidate_id"],
                    "dataset_id": row["dataset_id"],
                    "clip_id": row["clip_id"],
                    "frame_id": row["frame_id"],
                    "validation_role": "background",
                    "condition_group": scene_bucket or row["source_pool"],
                    "image_path": row["image_path"],
                }
            )
    return clean_rows, validation_rows, blockers


def materialize_dataset(positive_rows: list[dict], negative_rows: list[dict]) -> dict[str, dict[str, int]]:
    for generated_dir in ("yolo_dataset", "manifests", "reports", "cleaned_coco"):
        target = OUTPUT_ROOT / generated_dir
        if target.exists():
            shutil.rmtree(target)
    ensure_dir(REVIEWS_ROOT)
    dataset_root = ensure_dir(OUTPUT_ROOT / "yolo_dataset")
    ensure_dir(OUTPUT_ROOT / "manifests")
    ensure_dir(OUTPUT_ROOT / "reports")
    ensure_dir(OUTPUT_ROOT / "cleaned_coco")
    stats = {split: {"images": 0, "positives": 0, "negatives": 0, "boxes": 0} for split in ("train", "valid", "test")}

    coco_by_split = {
        split: {
            "info": {"description": "annotation_automation_v3 local-night corpus", "version": "3.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "streetlight", "supercategory": "light_source"}],
        }
        for split in ("train", "valid", "test")
    }
    annotation_rows: list[dict] = []
    image_id = 1
    annotation_id = 1

    for row in positive_rows:
        split = row["assigned_split"]
        image_src = Path(row["image_path"])
        image_dst = dataset_root / "images" / split / row["output_file_name"]
        label_dst = dataset_root / "labels" / split / f"{Path(row['output_file_name']).stem}.txt"
        copy_image(image_src, image_dst)
        ensure_dir(label_dst.parent)
        label_dst.write_text("\n".join(yolo_line(box, row["width"], row["height"]) for box in row["boxes"]) + "\n", encoding="utf-8")

        coco_by_split[split]["images"].append(
            {"id": image_id, "license": 1, "file_name": row["output_file_name"], "height": row["height"], "width": row["width"]}
        )
        for box in row["boxes"]:
            coco_by_split[split]["annotations"].append(
                {"id": annotation_id, "image_id": image_id, "category_id": 1, "bbox": box, "iscrowd": 0, "area": box[2] * box[3], "segmentation": []}
            )
            annotation_rows.append(
                {
                    "annotation_id": f"v3_{annotation_id}",
                    "image_uid": row["image_uid"],
                    "dataset_id": row["dataset_id"],
                    "clip_id": row["clip_id"],
                    "frame_id": row["frame_id"],
                    "image_path": row["image_path"],
                    "class_name": "streetlight",
                    "bbox_x": box[0],
                    "bbox_y": box[1],
                    "bbox_w": box[2],
                    "bbox_h": box[3],
                    "annotation_origin": row["corpus_role"],
                    "scene_bucket": row["scene_bucket"],
                    "review_status": row["review_status"],
                }
            )
            annotation_id += 1

        stats[split]["images"] += 1
        stats[split]["positives"] += 1
        stats[split]["boxes"] += len(row["boxes"])
        image_id += 1

    for row in negative_rows:
        split = row["assigned_split"]
        image_src = Path(row["image_path"])
        image_dst = dataset_root / "images" / split / row["output_file_name"]
        label_dst = dataset_root / "labels" / split / f"{Path(row['output_file_name']).stem}.txt"
        copy_image(image_src, image_dst)
        ensure_dir(label_dst.parent)
        label_dst.write_text("", encoding="utf-8")
        stats[split]["images"] += 1
        stats[split]["negatives"] += 1

    output_dataset_yaml(dataset_root)
    write_csv(
        OUTPUT_ROOT / "manifests" / "annotation_metadata_v3.csv",
        annotation_rows,
        ["annotation_id", "image_uid", "dataset_id", "clip_id", "frame_id", "image_path", "class_name", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "annotation_origin", "scene_bucket", "review_status"],
    )
    for split, coco in coco_by_split.items():
        write_json(OUTPUT_ROOT / "cleaned_coco" / f"{split}_annotations.coco.json", coco)
    return stats


def write_reports(
    positive_manifest_rows: list[dict],
    negative_manifest_rows: list[dict],
    validation_rows: list[dict],
    blockers: list[str],
    stats: dict[str, dict[str, int]] | None,
) -> None:
    ensure_dir(OUTPUT_ROOT / "reports")
    write_csv(
        OUTPUT_ROOT / "reports" / "validation_manifest_v3.csv",
        validation_rows,
        ["image_uid", "dataset_id", "clip_id", "frame_id", "validation_role", "condition_group", "image_path"],
    )

    merged_rows = positive_manifest_rows + negative_manifest_rows
    write_csv(
        OUTPUT_ROOT / "manifests" / "merged_manifest_v3.csv",
        merged_rows,
        ["image_uid", "dataset_id", "clip_id", "frame_id", "assigned_split", "corpus_role", "scene_bucket", "image_path", "output_file_name", "annotation_count", "review_status"],
    )

    clip_counts = Counter((row["dataset_id"], row["clip_id"], row["assigned_split"]) for row in positive_manifest_rows)
    scene_counts = Counter(row["scene_bucket"] or "scene_bucket_missing" for row in positive_manifest_rows + negative_manifest_rows)
    review_counts = Counter(row["review_status"] for row in positive_manifest_rows)
    negative_counts = Counter(row["review_status"] for row in negative_manifest_rows)

    lines = [
        "# Annotation Automation v3 Readiness Report",
        "",
        "## Positive review status",
        "",
    ]
    for status, count in sorted(review_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Negative review status", ""])
    for status, count in sorted(negative_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Scene bucket coverage", ""])
    for bucket, count in sorted(scene_counts.items()):
        lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(["", "## Clip assignments", "", "| Dataset | Clip | Split | Images |", "| --- | --- | --- | ---: |"])
    for (dataset_id, clip_id, split), count in sorted(clip_counts.items()):
        lines.append(f"| {dataset_id} | {clip_id} | {split} | {count} |")
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend([f"- {item}" for item in blockers[:200]])
        if len(blockers) > 200:
            lines.append(f"- ... {len(blockers) - 200} additional blockers omitted")
    else:
        lines.append("- none")
    if stats:
        lines.extend(["", "## Materialized stats", ""])
        for split in ("train", "valid", "test"):
            payload = stats[split]
            lines.append(
                f"- `{split}`: images=`{payload['images']}`, positives=`{payload['positives']}`, negatives=`{payload['negatives']}`, boxes=`{payload['boxes']}`"
            )

    (OUTPUT_ROOT / "reports" / "readiness_report_v3.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        OUTPUT_ROOT / "reports" / "corpus_summary_v3.json",
        {
            "positive_review_counts": dict(review_counts),
            "negative_review_counts": dict(negative_counts),
            "scene_bucket_counts": dict(scene_counts),
            "blocker_count": len(blockers),
            "materialized_stats": stats or {},
        },
    )


def main() -> None:
    args = parse_args()
    records, original_boxes = load_source_records()
    split_map = split_lookup()
    review_rows = review_lookup()
    negative_reviews = negative_review_lookup()
    existing_admissions = existing_negative_admissions()

    positive_rows, positive_manifest_rows, validation_positive_rows, positive_blockers = build_positive_rows(
        records,
        original_boxes,
        review_rows,
        split_map,
        args.allow_unreviewed_positives,
        args.allow_missing_scene_buckets,
    )
    negative_rows, validation_negative_rows, negative_blockers = build_negative_rows(
        negative_reviews,
        existing_admissions,
        args.allow_missing_scene_buckets,
        args.valid_negative_target,
        args.test_negative_target,
    )

    negative_manifest_rows = [
        {
            "image_uid": row["review_candidate_id"],
            "dataset_id": row["dataset_id"],
            "clip_id": row["clip_id"],
            "frame_id": row["frame_id"],
            "assigned_split": row["assigned_split"],
            "corpus_role": row["corpus_role"],
            "scene_bucket": row["scene_bucket"],
            "image_path": row["image_path"],
            "output_file_name": row["output_file_name"],
            "annotation_count": "0",
            "review_status": row["review_status"],
        }
        for row in negative_rows
    ]

    blockers = positive_blockers + negative_blockers
    stats = None
    if not blockers:
        stats = materialize_dataset(positive_rows, negative_rows)
    else:
        ensure_dir(OUTPUT_ROOT / "reports")
        ensure_dir(OUTPUT_ROOT / "manifests")

    write_reports(
        positive_manifest_rows,
        negative_manifest_rows,
        validation_positive_rows + validation_negative_rows,
        blockers,
        stats,
    )
    if blockers:
        print(json.dumps({"status": "blocked", "blocker_count": len(blockers)}, indent=2))
    else:
        print(json.dumps({"status": "materialized", "output_root": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
