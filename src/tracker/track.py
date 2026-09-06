"""Short-lived state for a single custom streetlight track."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TrackerConfig
from .ego_motion import transform_points
from .kalman import CentreScaleKalmanFilter


@dataclass
class TrackerTrack:
    track_id: int
    bbox: list[float]
    confidence: float
    class_id: int
    label: str
    centre: np.ndarray
    size: np.ndarray
    config: TrackerConfig
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    size_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    kalman_filter: CentreScaleKalmanFilter | None = None
    predicted_centre: np.ndarray = field(init=False)
    predicted_size: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.predicted_centre = self.centre.copy()
        self.predicted_size = self.size.copy()
        if self.config.uses_kalman:
            self.kalman_filter = CentreScaleKalmanFilter(
                np.concatenate([self.centre, self.size]),
                self.config.kalman_process_noise,
                self.config.kalman_measurement_noise,
            )

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.config.min_hits

    def predict(self, ego_transform: np.ndarray | None = None) -> None:
        self.age += 1
        self.time_since_update += 1
        if self.kalman_filter is not None:
            state = self.kalman_filter.predict()
            centre, size = state[:2], state[2:]
        else:
            centre, size = self.centre + self.velocity, self.size + self.size_velocity
        self.predicted_centre = np.asarray(centre, dtype=float)
        self.predicted_size = np.maximum(np.asarray(size, dtype=float), 1e-6)
        if ego_transform is not None:
            self.predicted_centre = transform_points(self.predicted_centre[None, :], ego_transform)[0]

    def update(self, detection: object) -> None:
        new_centre = np.asarray(getattr(detection, "centre"), dtype=float)
        new_size = np.asarray(getattr(detection, "size"), dtype=float)
        self.velocity = new_centre - self.centre
        self.size_velocity = new_size - self.size
        self.centre = new_centre
        self.size = new_size
        self.bbox = list(getattr(detection, "bbox"))
        self.confidence = float(getattr(detection, "confidence"))
        self.hits += 1
        self.time_since_update = 0
        if self.kalman_filter is not None:
            state = self.kalman_filter.update(np.concatenate([new_centre, new_size]))
            self.centre, self.size = state[:2], np.maximum(state[2:], 1e-6)
        self.predicted_centre = self.centre.copy()
        self.predicted_size = self.size.copy()

    def mark_missed(self) -> None:
        # State was already predicted; this method makes lifecycle intent clear.
        return None
