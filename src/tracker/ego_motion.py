"""Minimal global image-motion compensation for the ego-motion ablation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EgoMotionResult:
    transform: np.ndarray
    success: bool
    correspondences: int = 0
    inliers: int = 0
    reason: str | None = None


def identity_result(reason: str | None = None) -> EgoMotionResult:
    return EgoMotionResult(np.eye(3, dtype=float), False, reason=reason)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 2D affine/homogeneous image transform to point coordinates."""

    points = np.asarray(points, dtype=float).reshape((-1, 2))
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return points.copy()
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous @ matrix.T
    denominator = transformed[:, 2:3]
    valid = np.abs(denominator[:, 0]) > 1e-8
    output = points.copy()
    output[valid] = transformed[valid, :2] / denominator[valid]
    return output


class SparseEgoMotionEstimator:
    """Estimate only frame-to-frame *camera* motion using sparse LK + RANSAC.

    This is not object motion or visual odometry. Failure deliberately yields
    identity so the fourth ablation remains robust and interpretable.
    """

    def estimate(self, previous_frame: np.ndarray | None, current_frame: np.ndarray | None) -> EgoMotionResult:
        if previous_frame is None or current_frame is None:
            return identity_result("no_previous_frame")
        try:
            import cv2
        except ImportError:
            return identity_result("opencv_unavailable")
        try:
            previous_gray = self._gray(previous_frame, cv2)
            current_gray = self._gray(current_frame, cv2)
            previous_points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=300, qualityLevel=0.01, minDistance=7)
            if previous_points is None or len(previous_points) < 6:
                return identity_result("insufficient_features")
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_points, None)
            if current_points is None or status is None:
                return identity_result("feature_tracking_failed")
            valid = status.reshape(-1).astype(bool)
            source = previous_points.reshape(-1, 2)[valid]
            target = current_points.reshape(-1, 2)[valid]
            if len(source) < 6:
                return identity_result("insufficient_correspondences")
            affine, inlier_mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if affine is None or not np.isfinite(affine).all():
                return identity_result("ransac_failed")
            inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            if inliers < 4:
                return identity_result("insufficient_ransac_inliers")
            transform = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(float)
            return EgoMotionResult(transform, True, len(source), inliers)
        except Exception as error:  # OpenCV failures must never end inference.
            LOGGER.debug("Ego-motion estimation failed: %s", error)
            return identity_result("estimation_exception")

    @staticmethod
    def _gray(frame: np.ndarray, cv2: object) -> np.ndarray:
        image = np.asarray(frame)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
