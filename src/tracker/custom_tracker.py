"""Configurable centre-based streetlight association tracker."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .association import AssociationDetail, build_cost_matrix, hungarian_matches
from .config import TrackerConfig
from .ego_motion import EgoMotionResult, SparseEgoMotionEstimator, identity_result
from .geometry import is_valid_xywh, xywh_to_centre_scale
from .track import TrackerTrack


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackerDetection:
    bbox: list[float]
    confidence: float
    class_id: int
    label: str
    centre: np.ndarray
    size: np.ndarray
    source: Any


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    bbox: list[float]
    confidence: float
    class_id: int
    label: str
    age: int
    hits: int
    confirmed: bool
    association_cost: float | None
    predicted_centre: list[float]
    predicted_size: list[float]


class CustomStreetlightTracker:
    """Four ablations sharing lifecycle and centre/scale association logic."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: list[TrackerTrack] = []
        self._next_id = 1
        self._previous_frame: np.ndarray | None = None
        self._ego_motion = SparseEgoMotionEstimator() if self.config.uses_ego_motion else None
        self.diagnostics: list[dict[str, Any]] = []

    def update(self, detections: Sequence[object], frame: np.ndarray | None = None, frame_id: int | None = None) -> list[TrackedDetection]:
        """Associate one frame of project ``Detection`` instances.

        Only matching/new observations are emitted. Temporarily lost tracks
        remain internal until ``max_age`` is exceeded, preserving IDs after a
        short detector dropout without creating a second history store.
        """

        valid_detections = [item for item in (self._coerce_detection(det) for det in detections) if item is not None]
        ego = self._estimate_ego_motion(frame)
        for track in self._tracks:
            track.predict(ego.transform if self.config.uses_ego_motion else None)

        cost_matrix, details = build_cost_matrix(self._tracks, valid_detections, self.config)
        matches = hungarian_matches(cost_matrix)
        matched_track_indices = {track_index for track_index, _ in matches}
        matched_detection_indices = {detection_index for _, detection_index in matches}
        output: list[TrackedDetection] = []

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = valid_detections[detection_index]
            detail = details[track_index][detection_index]
            track.update(detection)
            output.append(self._observation(track, detail.cost))
            self._log_match(frame_id, track, detection_index, detail, ego, "matched")

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_track_indices:
                track.mark_missed()
                self._log_match(frame_id, track, None, None, ego, "unmatched_track")

        for detection_index, detection in enumerate(valid_detections):
            if detection_index in matched_detection_indices:
                continue
            track = self._new_track(detection)
            self._tracks.append(track)
            output.append(self._observation(track, None))
            self._log_match(frame_id, track, detection_index, None, ego, "new_track")

        self._tracks = [track for track in self._tracks if track.time_since_update <= self.config.max_age]
        self._previous_frame = None if frame is None else np.asarray(frame).copy()
        return output

    def _estimate_ego_motion(self, frame: np.ndarray | None) -> EgoMotionResult:
        if self._ego_motion is None:
            return identity_result("ego_motion_disabled")
        result = self._ego_motion.estimate(self._previous_frame, frame)
        if not result.success:
            LOGGER.debug("Ego-motion identity fallback: %s", result.reason)
        return result

    def _new_track(self, detection: TrackerDetection) -> TrackerTrack:
        track = TrackerTrack(
            track_id=self._next_id,
            bbox=list(detection.bbox),
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
            centre=detection.centre.copy(),
            size=detection.size.copy(),
            config=self.config,
        )
        self._next_id += 1
        return track

    @staticmethod
    def _coerce_detection(detection: object) -> TrackerDetection | None:
        bbox = list(getattr(detection, "bbox", []))
        if not is_valid_xywh(bbox):
            return None
        try:
            confidence = float(getattr(detection, "score", 0.0))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(confidence):
            return None
        metadata = getattr(detection, "metadata", {}) or {}
        class_id = int(metadata.get("class_id", 0))
        label = str(getattr(detection, "label", class_id))
        centre, size = xywh_to_centre_scale(bbox)
        return TrackerDetection([float(value) for value in bbox[:4]], confidence, class_id, label, centre, size, detection)

    @staticmethod
    def _observation(track: TrackerTrack, association_cost: float | None) -> TrackedDetection:
        return TrackedDetection(
            track.track_id,
            list(track.bbox),
            track.confidence,
            track.class_id,
            track.label,
            track.age,
            track.hits,
            track.confirmed,
            association_cost,
            track.predicted_centre.astype(float).tolist(),
            track.predicted_size.astype(float).tolist(),
        )

    def _log_match(self, frame_id: int | None, track: TrackerTrack, detection_index: int | None, detail: AssociationDetail | None, ego: EgoMotionResult, status: str) -> None:
        if not self.config.debug:
            return
        self.diagnostics.append(
            {
                "frame_id": frame_id,
                "track_id": track.track_id,
                "detection_index": detection_index,
                "status": status,
                "association_cost": None if detail is None else detail.cost,
                "centre_distance": None if detail is None else detail.centre_distance,
                "scale_distance": None if detail is None else detail.scale_distance,
                "track_age": track.age,
                "time_since_update": track.time_since_update,
                "predicted_centre": track.predicted_centre.astype(float).tolist(),
                "predicted_size": track.predicted_size.astype(float).tolist(),
                "ego_motion_success": ego.success,
                "ego_motion_correspondences": ego.correspondences,
                "ego_motion_inliers": ego.inliers,
                "ego_motion_reason": ego.reason,
            }
        )
