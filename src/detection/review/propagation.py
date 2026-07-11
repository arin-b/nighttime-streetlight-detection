from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


from rbccps_od.config.paths import repo_root


ROOT = repo_root()
APP_REVIEWS_ROOT = ROOT / "datasets" / "derived" / "annotation_click_review" / "reviews"
V3_REVIEWS_ROOT = ROOT / "datasets" / "derived" / "annotation_automation_v3" / "reviews"
REPORTS_ROOT = V3_REVIEWS_ROOT / "propagation_reports"

MAX_INTERPOLATION_GAP = 8


@dataclass
class Candidate:
    mode: str
    key: str
    image_uid: str
    dataset_id: str
    clip_id: str
    frame_id: str
    image_path: str
    annotation_count: str
    current_split_v2: str
    primary_decision: str
    secondary_reason: str
    scene_bucket: str
    rule: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def frame_num(row: dict[str, str]) -> int:
    match = re.search(r"(\d+)$", row["frame_id"])
    return int(match.group(1)) if match else -1


def convert_status_to_app_fields(review_status: str) -> tuple[str, str]:
    if review_status == "keep":
        return "keep", ""
    if review_status.startswith("exclude_"):
        return "exclude", review_status
    raise ValueError(f"Unsupported propagated review status: {review_status}")


def build_candidates(v3_rows: list[dict[str, str]], mode: str) -> list[Candidate]:
    by_clip: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in v3_rows:
        by_clip[row["clip_id"]].append(row)

    candidates: list[Candidate] = []
    for clip_id, clip_rows in by_clip.items():
        items = sorted(clip_rows, key=frame_num)
        reviewed = [row for row in items if row["review_status"] != "pending"]
        if not reviewed:
            continue

        statuses = {row["review_status"] for row in reviewed}
        scenes = {row["scene_bucket"] for row in reviewed if row["scene_bucket"]}
        unanimous_status = len(statuses) == 1
        unanimous_scene = len(scenes) == 1
        unanimous_value = next(iter(statuses))
        unanimous_scene_value = next(iter(scenes)) if scenes else ""

        for idx, row in enumerate(items):
            if row["review_status"] != "pending":
                continue

            propagated_status = ""
            propagated_scene = ""
            rule = ""

            if unanimous_status and unanimous_value != "fix_box":
                if unanimous_value == "keep":
                    if not unanimous_scene:
                        continue
                    propagated_status = unanimous_value
                    propagated_scene = unanimous_scene_value
                    rule = "clip_unanimous_keep"
                elif unanimous_value.startswith("exclude_"):
                    propagated_status = unanimous_value
                    propagated_scene = unanimous_scene_value
                    rule = "clip_unanimous_exclude"
            else:
                prev_row = next(
                    (items[j] for j in range(idx - 1, -1, -1) if items[j]["review_status"] != "pending"),
                    None,
                )
                next_row = next(
                    (items[j] for j in range(idx + 1, len(items)) if items[j]["review_status"] != "pending"),
                    None,
                )
                if not prev_row or not next_row:
                    continue

                prev_status = prev_row["review_status"]
                next_status = next_row["review_status"]
                prev_scene = prev_row.get("scene_bucket", "").strip()
                next_scene = next_row.get("scene_bucket", "").strip()
                gap = frame_num(next_row) - frame_num(prev_row)

                if prev_status != next_status or prev_status == "fix_box" or gap > MAX_INTERPOLATION_GAP:
                    continue

                if prev_status == "keep":
                    if prev_scene and prev_scene == next_scene:
                        propagated_status = "keep"
                        propagated_scene = prev_scene
                        rule = "between_matching_keep_anchors"
                elif prev_status.startswith("exclude_"):
                    propagated_status = prev_status
                    propagated_scene = prev_scene if prev_scene == next_scene else ""
                    rule = "between_matching_exclude_anchors"

            if not propagated_status:
                continue

            primary_decision, secondary_reason = convert_status_to_app_fields(propagated_status)
            candidates.append(
                Candidate(
                    mode=mode,
                    key=row["key"],
                    image_uid=row["image_uid"],
                    dataset_id=row["dataset_id"],
                    clip_id=row["clip_id"],
                    frame_id=row["frame_id"],
                    image_path=row["image_path"],
                    annotation_count=row["annotation_count"],
                    current_split_v2=row["current_split_v2"],
                    primary_decision=primary_decision,
                    secondary_reason=secondary_reason,
                    scene_bucket=propagated_scene,
                    rule=rule,
                )
            )

    return candidates


def merge_into_app_csv(app_csv: Path, candidates: list[Candidate]) -> tuple[int, int]:
    rows = load_csv(app_csv)
    if not rows:
        raise RuntimeError(f"Expected existing app review CSV at {app_csv}")
    fieldnames = list(rows[0].keys())
    existing_keys = {row["key"] for row in rows}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    added = 0
    skipped_existing = 0
    for candidate in candidates:
        if candidate.key in existing_keys:
            skipped_existing += 1
            continue
        rows.append(
            {
                "key": candidate.key,
                "image_uid": candidate.image_uid,
                "dataset_id": candidate.dataset_id,
                "clip_id": candidate.clip_id,
                "frame_id": candidate.frame_id,
                "image_path": candidate.image_path,
                "annotation_count": candidate.annotation_count,
                "current_split_v2": candidate.current_split_v2,
                "primary_decision": candidate.primary_decision,
                "secondary_reason": candidate.secondary_reason,
                "scene_bucket": candidate.scene_bucket,
                "updated_boxes_json": "",
                "review_timestamp": now,
            }
        )
        existing_keys.add(candidate.key)
        added += 1

    write_csv(app_csv, rows, fieldnames)
    return added, skipped_existing


def write_report(mode: str, candidates: list[Candidate]) -> None:
    ensure_dir(REPORTS_ROOT)
    report_path = REPORTS_ROOT / f"{mode}_auto_propagation.csv"
    rows = [
        {
            "mode": candidate.mode,
            "key": candidate.key,
            "clip_id": candidate.clip_id,
            "frame_id": candidate.frame_id,
            "primary_decision": candidate.primary_decision,
            "secondary_reason": candidate.secondary_reason,
            "scene_bucket": candidate.scene_bucket,
            "rule": candidate.rule,
        }
        for candidate in candidates
    ]
    write_csv(
        report_path,
        rows,
        ["mode", "key", "clip_id", "frame_id", "primary_decision", "secondary_reason", "scene_bucket", "rule"],
    )


def main() -> None:
    per_mode = {
        "jobin_positive": (
            V3_REVIEWS_ROOT / "jobin_positive_review_v3.csv",
            APP_REVIEWS_ROOT / "jobin_positive.csv",
        ),
        "arindam_positive": (
            V3_REVIEWS_ROOT / "arindam_positive_review_v3.csv",
            APP_REVIEWS_ROOT / "arindam_positive.csv",
        ),
    }

    summary: dict[str, object] = {}
    for mode, (v3_csv, app_csv) in per_mode.items():
        candidates = build_candidates(load_csv(v3_csv), mode)
        added, skipped_existing = merge_into_app_csv(app_csv, candidates)
        write_report(mode, candidates)
        decision_counts = Counter(
            candidate.secondary_reason if candidate.primary_decision == "exclude" else candidate.primary_decision
            for candidate in candidates
        )
        summary[mode] = {
            "candidates_total": len(candidates),
            "rows_added": added,
            "rows_skipped_existing": skipped_existing,
            "decision_counts": dict(decision_counts),
        }

    ensure_dir(REPORTS_ROOT)
    (REPORTS_ROOT / "auto_propagation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
