from __future__ import annotations


def timestamp_delta_ns(left_ns: int, right_ns: int) -> int:
    return int(left_ns) - int(right_ns)
