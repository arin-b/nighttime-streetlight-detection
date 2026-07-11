from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from rbccps_od.config.paths import datasets_root


ROOT = datasets_root()
OUTPUT_ROOT = ROOT / "derived" / "annotation_automation"
DEFAULT_REVIEW_MANIFEST = OUTPUT_ROOT / "reviews" / "hard_negative_review_manifest.csv"
DEFAULT_MERGED_MANIFEST = OUTPUT_ROOT / "manifests" / "merged_manifest.csv"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "yolo_dataset"

ALLOWED_REVIEW_LABELS = {"pending", "clean_negative", "ambiguous", "missed_positive"}
STABLE_SPLIT_BANDS = (("train", 0.70), ("valid", 0.85), ("test", 1.00))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Admit reviewed negative frames into the derived YOLO corpus.")
    parser.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST))
    parser.add_argument("--merged-manifest", default=str(DEFAULT_MERGED_MANIFEST))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT / "integrated_reviews"))
    parser.add_argument("--strict", action="store_true", help="Fail if pending review rows still exist.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def deterministic_split(dataset_id: str, clip_id: str, frame_id: str) -> str:
    token = f"{dataset_id}|{clip_id}|{frame_id}".encode("utf-8")
    ratio = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
    for split_name, upper in STABLE_SPLIT_BANDS:
        if ratio <= upper:
            return split_name
    return "test"


def build_known_split_map(rows: list[dict]) -> dict[tuple[str, str], str]:
    split_map: dict[tuple[str, str], str] = {}
    for row in rows:
        dataset_id = row.get("dataset_id", "")
        clip_id = row.get("clip_id", "")
        assigned_split = row.get("assigned_split", "")
        if dataset_id and clip_id and assigned_split:
            split_map[(dataset_id, clip_id)] = assigned_split
    return split_map


def copy_negative_example(src: Path, dataset_root: Path, split_name: str, output_stem: str) -> tuple[Path, Path]:
    image_dst = dataset_root / "images" / split_name / f"{output_stem}{src.suffix.lower()}"
    label_dst = dataset_root / "labels" / split_name / f"{output_stem}.txt"
    ensure_dir(image_dst.parent)
    ensure_dir(label_dst.parent)
    if not image_dst.exists():
        shutil.copy2(src, image_dst)
    label_dst.write_text("", encoding="utf-8")
    return image_dst, label_dst


def main() -> None:
    args = parse_args()
    review_rows = load_rows(Path(args.review_manifest))
    merged_rows = load_rows(Path(args.merged_manifest))
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    known_split_map = build_known_split_map(merged_rows)

    invalid_labels = sorted({row.get("review_label", "") for row in review_rows if row.get("review_label", "") not in ALLOWED_REVIEW_LABELS})
    if invalid_labels:
        raise SystemExit(f"Invalid review labels found: {invalid_labels}")

    pending_rows = [row for row in review_rows if row.get("review_label", "") == "pending"]
    if args.strict and pending_rows:
        raise SystemExit(f"Strict mode enabled, but {len(pending_rows)} review rows are still pending.")

    admitted_rows: list[dict] = []
    ambiguous_rows: list[dict] = []
    missed_positive_rows: list[dict] = []
    skipped_pending_rows: list[dict] = []

    for row in review_rows:
        review_label = row.get("review_label", "")
        if review_label == "pending":
            skipped_pending_rows.append(row)
            continue
        if review_label == "ambiguous":
            ambiguous_rows.append(row)
            continue
        if review_label == "missed_positive":
            missed_positive_rows.append(row)
            continue

        dataset_id = row.get("dataset_id", "")
        clip_id = row.get("clip_id", "")
        frame_id = row.get("frame_id", "")
        split_name = known_split_map.get((dataset_id, clip_id)) or deterministic_split(dataset_id, clip_id, frame_id)
        image_path = Path(row["image_path"])
        output_stem = f"neg__{row['review_candidate_id']}__{image_path.stem}"
        image_dst, label_dst = copy_negative_example(image_path, dataset_root, split_name, output_stem)
        admitted_rows.append(
            {
                "review_candidate_id": row["review_candidate_id"],
                "dataset_id": dataset_id,
                "clip_id": clip_id,
                "frame_id": frame_id,
                "source_pool": row.get("source_pool", ""),
                "image_path": str(image_path),
                "assigned_split": split_name,
                "copied_image_path": str(image_dst),
                "label_path": str(label_dst),
                "suggested_negative_subtype": row.get("suggested_negative_subtype", ""),
                "review_label": review_label,
                "notes": row.get("notes", ""),
            }
        )

    review_fields = list(review_rows[0].keys()) if review_rows else []
    write_csv(
        output_root / "reviewed_negative_admissions.csv",
        admitted_rows,
        [
            "review_candidate_id",
            "dataset_id",
            "clip_id",
            "frame_id",
            "source_pool",
            "image_path",
            "assigned_split",
            "copied_image_path",
            "label_path",
            "suggested_negative_subtype",
            "review_label",
            "notes",
        ],
    )
    write_csv(output_root / "ambiguous_review_rows.csv", ambiguous_rows, review_fields)
    write_csv(output_root / "missed_positive_review_rows.csv", missed_positive_rows, review_fields)
    write_csv(output_root / "pending_review_rows.csv", skipped_pending_rows, review_fields)

    print(f"Admitted clean negatives: {len(admitted_rows)}")
    print(f"Ambiguous rows: {len(ambiguous_rows)}")
    print(f"Missed-positive rows: {len(missed_positive_rows)}")
    print(f"Pending rows: {len(skipped_pending_rows)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
