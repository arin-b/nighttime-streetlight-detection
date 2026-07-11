from __future__ import annotations

from rbccps_od.data_management.corpus_v3 import build_negative_rows

__all__ = ["build_negative_rows", "is_retained_negative", "is_promoted_negative"]

NEGATIVE_RETAINED_STATUSES = {"clean_negative"}
NEGATIVE_PROMOTED_STATUSES = {"promote_to_positive_review"}


def is_retained_negative(status: str) -> bool:
    return status in NEGATIVE_RETAINED_STATUSES


def is_promoted_negative(status: str) -> bool:
    return status in NEGATIVE_PROMOTED_STATUSES
