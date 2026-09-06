"""A small constant-velocity Kalman filter for centre and box scale."""

from __future__ import annotations

import numpy as np


class CentreScaleKalmanFilter:
    """State: ``[cx, cy, w, h, vx, vy, vw, vh]`` with unit frame timestep."""

    def __init__(self, measurement: np.ndarray, process_noise: float, measurement_noise: float) -> None:
        measurement = np.asarray(measurement, dtype=float)
        if measurement.shape != (4,) or not np.isfinite(measurement).all():
            raise ValueError("Kalman measurement must contain four finite values.")
        self.state = np.concatenate([measurement, np.zeros(4, dtype=float)])
        self.transition = np.eye(8, dtype=float)
        self.transition[:4, 4:] = np.eye(4, dtype=float)
        self.observation = np.zeros((4, 8), dtype=float)
        self.observation[:, :4] = np.eye(4, dtype=float)
        self.covariance = np.eye(8, dtype=float)
        self.process_noise = np.eye(8, dtype=float) * float(process_noise)
        self.measurement_noise = np.eye(4, dtype=float) * float(measurement_noise)

    def predict(self) -> np.ndarray:
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        return self.state[:4].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        measurement = np.asarray(measurement, dtype=float)
        innovation = measurement - self.observation @ self.state
        innovation_covariance = self.observation @ self.covariance @ self.observation.T + self.measurement_noise
        gain = self.covariance @ self.observation.T @ np.linalg.pinv(innovation_covariance)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(8) - gain @ self.observation) @ self.covariance
        return self.state[:4].copy()
