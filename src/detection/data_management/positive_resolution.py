from __future__ import annotations

from rbccps_od.data_management.corpus_v3 import build_positive_rows

__all__ = ["build_positive_rows", "is_retained_positive", "is_excluded_positive"]

RETAINED_POSITIVE_STATUSES = {"keep", "fix_box"}


def is_retained_positive(status: str) -> bool:
    return status in RETAINED_POSITIVE_STATUSES


def is_excluded_positive(status: str) -> bool:
    return status.startswith("exclude_")
