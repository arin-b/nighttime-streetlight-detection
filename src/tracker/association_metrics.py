"""Offline, manually-labelled association metric experiment helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping

from .geometry import EPSILON, centre_distance, iou_xyxy, xyxy_to_centre_scale


def pair_metrics(first: Mapping[str, object], second: Mapping[str, object]) -> dict[str, float]:
    first_box = [float(first[key]) for key in ("x1", "y1", "x2", "y2")]
    second_box = [float(second[key]) for key in ("x1", "y1", "x2", "y2")]
    first_centre, first_size = xyxy_to_centre_scale(first_box)
    second_centre, second_size = xyxy_to_centre_scale(second_box)
    distance = centre_distance(first_centre, second_centre)
    area = max(float(first_size[0] * first_size[1]), float(second_size[0] * second_size[1]))
    overlap = iou_xyxy(first_box, second_box)
    return {
        "centre_distance": distance,
        "normalized_centre_distance": distance / math.sqrt(max(area, EPSILON)),
        "iou": overlap,
        "one_minus_iou": 1.0 - overlap,
    }


def consecutive_positive_pairs(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["lamp_id"])].append(row)
    results: list[dict[str, object]] = []
    for lamp_id, observations in grouped.items():
        ordered = sorted(observations, key=lambda item: int(item["frame"]))
        for first, second in zip(ordered, ordered[1:]):
            if int(second["frame"]) != int(first["frame"]) + 1:
                continue
            results.append({"pair_type": "positive", "lamp_id": lamp_id, "frame_t": int(first["frame"]), "frame_t1": int(second["frame"])} | pair_metrics(first, second))
    return results


def consecutive_negative_pairs(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Optional all different-lamp pairs across neighbouring frames."""

    by_frame: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    results: list[dict[str, object]] = []
    for frame, observations in by_frame.items():
        for first in observations:
            for second in by_frame.get(frame + 1, []):
                if str(first["lamp_id"]) != str(second["lamp_id"]):
                    results.append({"pair_type": "negative", "lamp_id": f"{first['lamp_id']}->{second['lamp_id']}", "frame_t": frame, "frame_t1": frame + 1} | pair_metrics(first, second))
    return results
