from __future__ import annotations

from collections import defaultdict

from rbccps_od.data_management.seed_sources import assign_clip_splits, choose_split_counts

__all__ = [
    "assign_clip_splits",
    "choose_split_counts",
    "ensure_clip_held_out",
    "split_counts_by_clip",
]


def ensure_clip_held_out(rows: list[dict], split_field: str = "assigned_split", clip_field: str = "clip_id") -> dict[str, str]:
    clip_to_split: dict[str, str] = {}
    for row in rows:
        clip_id = row.get(clip_field, "")
        split = row.get(split_field, "")
        if not clip_id or not split:
            continue
        existing = clip_to_split.get(clip_id)
        if existing and existing != split:
            raise ValueError(f"Clip leakage detected for {clip_id}: {existing} vs {split}")
        clip_to_split[clip_id] = split
    return clip_to_split


def split_counts_by_clip(rows: list[dict], split_field: str = "assigned_split", clip_field: str = "clip_id") -> dict[str, int]:
    clip_splits = ensure_clip_held_out(rows, split_field=split_field, clip_field=clip_field)
    counts: dict[str, int] = defaultdict(int)
    for split in clip_splits.values():
        counts[split] += 1
    return dict(counts)
