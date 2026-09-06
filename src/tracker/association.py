"""Centre/scale-only association; IoU is intentionally absent here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .config import TrackerConfig
from .geometry import centre_distance, logarithmic_scale_distance


INVALID_COST = float("inf")


class PredictableTrack(Protocol):
    class_id: int
    predicted_centre: np.ndarray
    predicted_size: np.ndarray


@dataclass(frozen=True)
class AssociationDetail:
    cost: float
    centre_distance: float
    scale_distance: float | None


def association_detail(track: PredictableTrack, detection: object, config: TrackerConfig) -> AssociationDetail:
    """Return a gated association cost without any IoU or appearance term."""

    if track.class_id != int(getattr(detection, "class_id")):
        return AssociationDetail(INVALID_COST, float("inf"), None)
    detection_centre = np.asarray(getattr(detection, "centre"), dtype=float)
    centre_cost = centre_distance(track.predicted_centre, detection_centre)
    if not np.isfinite(centre_cost) or centre_cost > config.max_center_distance:
        return AssociationDetail(INVALID_COST, centre_cost, None)
    if not config.uses_scale:
        return AssociationDetail(config.center_weight * centre_cost, centre_cost, None)
    scale_cost = logarithmic_scale_distance(track.predicted_size, np.asarray(getattr(detection, "size"), dtype=float))
    if not np.isfinite(scale_cost) or scale_cost > config.max_scale_distance:
        return AssociationDetail(INVALID_COST, centre_cost, scale_cost)
    return AssociationDetail(
        config.center_weight * centre_cost + config.scale_weight * scale_cost,
        centre_cost,
        scale_cost,
    )


def build_cost_matrix(tracks: Sequence[PredictableTrack], detections: Sequence[object], config: TrackerConfig) -> tuple[np.ndarray, list[list[AssociationDetail]]]:
    details = [[association_detail(track, detection, config) for detection in detections] for track in tracks]
    return np.asarray([[detail.cost for detail in row] for row in details], dtype=float).reshape((len(tracks), len(detections))), details


def hungarian_matches(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return only finite one-to-one assignments from a gated cost matrix."""

    if cost_matrix.size == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    finite = np.isfinite(cost_matrix)
    if not finite.any():
        return []
    safe_cost = np.where(finite, cost_matrix, 1e12)
    rows, columns = linear_sum_assignment(safe_cost)
    return [(int(row), int(column)) for row, column in zip(rows, columns) if finite[row, column]]
