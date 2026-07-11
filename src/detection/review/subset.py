from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


from rbccps_od.config.paths import repo_root


ROOT = repo_root()
DATASETS_ROOT = ROOT / "datasets"
V3_REVIEWS_ROOT = DATASETS_ROOT / "derived" / "annotation_automation_v3" / "reviews"
CLICK_REVIEWS_ROOT = DATASETS_ROOT / "derived" / "annotation_click_review" / "reviews"
FRAME_RE = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a thinned review subset for repetitive pending frames.")
    parser.add_argument(
        "--mode",
        default="jobin_positive",
        choices=["jobin_positive", "arindam_positive", "negative_review"],
        help="Review mode to thin.",
    )
    parser.add_argument("--stride", type=int, default=4, help="Keep every Nth pending frame within a clip.")
    parser.add_argument("--neighbor-window", type=int, default=2, help="Always keep pending frames within this frame distance of any reviewed frame.")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frame_num(frame_id: str) -> int:
    match = FRAME_RE.search(frame_id or "")
    return int(match.group(1)) if match else -1


def main() -> None:
    args = parse_args()
    source_name = "negative_review_v3.csv" if args.mode == "negative_review" else f"{args.mode}_review_v3.csv"
    source_path = V3_REVIEWS_ROOT / source_name
    rows = load_csv(source_path)

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_clip[row["clip_id"]].append(row)
    for clip_rows in by_clip.values():
        clip_rows.sort(key=lambda row: (frame_num(row["frame_id"]), row["key"]))

    selected_keys: list[str] = []
    preview_rows: list[dict] = []
    total_pending = 0

    for clip_id, clip_rows in sorted(by_clip.items()):
        reviewed_frames = [frame_num(row["frame_id"]) for row in clip_rows if row["review_status"] != "pending"]
        pending_rows = [row for row in clip_rows if row["review_status"] == "pending"]
        total_pending += len(pending_rows)
        for index, row in enumerate(pending_rows):
            fnum = frame_num(row["frame_id"])
            keep = False
            reason = ""
            if index == 0:
                keep = True
                reason = "first_pending_in_clip"
            elif index == len(pending_rows) - 1:
                keep = True
                reason = "last_pending_in_clip"
            elif args.stride > 0 and index % args.stride == 0:
                keep = True
                reason = "stride_sample"
            elif any(abs(fnum - reviewed) <= args.neighbor_window for reviewed in reviewed_frames if reviewed >= 0 and fnum >= 0):
                keep = True
                reason = "near_reviewed_anchor"

            if keep:
                selected_keys.append(row["key"])
                preview_rows.append(
                    {
                        "key": row["key"],
                        "clip_id": clip_id,
                        "frame_id": row["frame_id"],
                        "review_status": row["review_status"],
                        "selection_reason": reason,
                    }
                )

    subset_path = CLICK_REVIEWS_ROOT / f"{args.mode}_subset_keys.txt"
    subset_path.write_text("\n".join(selected_keys) + ("\n" if selected_keys else ""), encoding="utf-8")

    preview_path = CLICK_REVIEWS_ROOT / f"{args.mode}_subset_preview.csv"
    with preview_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "clip_id", "frame_id", "review_status", "selection_reason"])
        writer.writeheader()
        writer.writerows(preview_rows)

    summary = {
        "mode": args.mode,
        "source_review_file": str(source_path),
        "subset_keys_file": str(subset_path),
        "subset_preview_file": str(preview_path),
        "pending_total": total_pending,
        "subset_selected": len(selected_keys),
        "stride": args.stride,
        "neighbor_window": args.neighbor_window,
    }
    summary_path = CLICK_REVIEWS_ROOT / f"{args.mode}_subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
