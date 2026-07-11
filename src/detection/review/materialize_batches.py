from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from rbccps_od.config.paths import datasets_root


ROOT = datasets_root()
DEFAULT_OUTPUT_ROOT = ROOT / "derived" / "annotation_automation" / "reviews" / "batches"
DEFAULT_HARD_NEGATIVE_MANIFEST = ROOT / "derived" / "annotation_automation" / "reviews" / "hard_negative_review_manifest.csv"
DEFAULT_CALIBRATION_MANIFEST = ROOT / "derived" / "annotation_automation" / "reviews" / "calibration_subset_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy review images into reviewer-friendly batch folders.")
    parser.add_argument("--hard-negative-manifest", default=str(DEFAULT_HARD_NEGATIVE_MANIFEST))
    parser.add_argument("--calibration-manifest", default=str(DEFAULT_CALIBRATION_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    cleaned = value.strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in cleaned)


def copy_if_needed(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)


def materialize_hard_negatives(rows: list[dict], output_root: Path) -> int:
    batch_root = ensure_dir(output_root / "hard_negative")
    written = 0
    for row in rows:
        image_path = Path(row["image_path"])
        subtype = safe_name(row.get("suggested_negative_subtype") or "unsorted")
        priority = safe_name(row.get("priority") or "unspecified")
        file_name = f"{row['review_candidate_id']}__{image_path.name}"
        dst = batch_root / priority / subtype / file_name
        copy_if_needed(image_path, dst)
        written += 1
    return written


def materialize_calibration(rows: list[dict], output_root: Path) -> int:
    batch_root = ensure_dir(output_root / "calibration")
    written = 0
    for row in rows:
        image_path = Path(row["image_path"])
        stratum = safe_name(row.get("stratum") or "unsorted")
        file_name = f"{row['calibration_id']}__{image_path.name}"
        dst = batch_root / stratum / file_name
        copy_if_needed(image_path, dst)
        written += 1
    return written


def write_index(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    hard_negative_rows = load_rows(Path(args.hard_negative_manifest))
    calibration_rows = load_rows(Path(args.calibration_manifest))

    hard_negative_count = materialize_hard_negatives(hard_negative_rows, output_root)
    calibration_count = materialize_calibration(calibration_rows, output_root)

    hard_fields = list(hard_negative_rows[0].keys()) if hard_negative_rows else []
    cal_fields = list(calibration_rows[0].keys()) if calibration_rows else []
    write_index(output_root / "hard_negative_batch_index.csv", hard_negative_rows, fieldnames=hard_fields)
    write_index(output_root / "calibration_batch_index.csv", calibration_rows, fieldnames=cal_fields)

    print(f"Materialized {hard_negative_count} hard-negative review images.")
    print(f"Materialized {calibration_count} calibration review images.")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
