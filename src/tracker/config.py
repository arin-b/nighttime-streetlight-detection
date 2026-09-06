"""Configuration for the controlled custom-tracker ablation."""

from __future__ import annotations

from dataclasses import dataclass


TRACKING_MODES = (
    "centre",
    "centre_scale",
    "centre_scale_kalman",
    "centre_scale_kalman_ego",
)


@dataclass(frozen=True)
class TrackerConfig:
    """Parameters shared by all four tracking modes.

    Distances are pixels and ``max_scale_distance`` is the L2 distance between
    logarithmic width/height ratios.
    """

    mode: str = "centre"
    max_age: int = 5
    min_hits: int = 3
    max_center_distance: float = 100.0
    max_scale_distance: float = 1.0
    center_weight: float = 1.0
    scale_weight: float = 0.25
    kalman_process_noise: float = 1.0
    kalman_measurement_noise: float = 10.0
    ego_motion_enabled: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        if self.mode not in TRACKING_MODES:
            raise ValueError(f"Unknown tracking mode {self.mode!r}; expected one of {TRACKING_MODES}.")
        if self.max_age < 0 or self.min_hits < 1:
            raise ValueError("max_age must be non-negative and min_hits must be at least one.")
        if self.max_center_distance <= 0 or self.max_scale_distance <= 0:
            raise ValueError("Association gates must be positive.")

    @property
    def uses_scale(self) -> bool:
        return self.mode != "centre"

    @property
    def uses_kalman(self) -> bool:
        return self.mode in {"centre_scale_kalman", "centre_scale_kalman_ego"}

    @property
    def uses_ego_motion(self) -> bool:
        return self.mode == "centre_scale_kalman_ego" and self.ego_motion_enabled
