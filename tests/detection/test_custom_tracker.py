from __future__ import annotations

import numpy as np

from rbccps_od.domain.detections import Detection
from tracker import CustomStreetlightTracker, TrackerConfig
from tracker.association import build_cost_matrix, hungarian_matches
from tracker.ego_motion import transform_points
from tracker.geometry import centre_distance, iou_xyxy, logarithmic_scale_distance, xyxy_to_centre_scale


def detection(box: list[float], *, class_id: int = 0) -> Detection:
    return Detection(bbox=box, score=0.9, label="streetlight", metadata={"class_id": class_id})


def test_xyxy_conversion_and_distances() -> None:
    centre, size = xyxy_to_centre_scale([0, 0, 10, 20])
    assert centre.tolist() == [5.0, 10.0]
    assert size.tolist() == [10.0, 20.0]
    assert centre_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0
    assert logarithmic_scale_distance(np.array([10.0, 10.0]), np.array([10.0, 10.0])) == 0.0
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_centre_tracker_uses_one_to_one_gated_assignment() -> None:
    tracker = CustomStreetlightTracker(TrackerConfig(mode="centre", max_center_distance=15.0))
    first = tracker.update([detection([10, 10, 4, 4]), detection([100, 100, 4, 4])])
    second = tracker.update([detection([12, 10, 4, 4]), detection([102, 100, 4, 4])])
    assert [item.track_id for item in first] == [1, 2]
    assert [item.track_id for item in second] == [1, 2]
    assert len({item.track_id for item in second}) == len(second)
    assert tracker.update([detection([300, 300, 4, 4])])[0].track_id == 3


def test_track_survives_miss_then_is_deleted_after_max_age() -> None:
    tracker = CustomStreetlightTracker(TrackerConfig(mode="centre", max_center_distance=20.0, max_age=1))
    assert tracker.update([detection([10, 10, 4, 4])])[0].track_id == 1
    assert tracker.update([]) == []
    assert tracker.update([detection([11, 10, 4, 4])])[0].track_id == 1
    tracker.update([])
    tracker.update([])
    assert tracker.update([detection([11, 10, 4, 4])])[0].track_id == 2


def test_scale_gate_and_class_gate_reject_candidates() -> None:
    tracker = CustomStreetlightTracker(TrackerConfig(mode="centre_scale", max_center_distance=20.0, max_scale_distance=0.2))
    assert tracker.update([detection([10, 10, 10, 10], class_id=0)])[0].track_id == 1
    assert tracker.update([detection([10, 10, 30, 30], class_id=0)])[0].track_id == 2
    assert tracker.update([detection([10, 10, 10, 10], class_id=1)])[0].track_id == 3


def test_cost_matrix_and_hungarian_matching_dimensions() -> None:
    tracker = CustomStreetlightTracker(TrackerConfig(mode="centre"))
    tracker.update([detection([10, 10, 4, 4]), detection([30, 30, 4, 4])])
    candidates = [tracker._coerce_detection(detection([11, 10, 4, 4])), tracker._coerce_detection(detection([31, 30, 4, 4]))]
    for track in tracker._tracks:
        track.predict()
    matrix, _ = build_cost_matrix(tracker._tracks, candidates, tracker.config)
    assert matrix.shape == (2, 2)
    assert hungarian_matches(matrix) == [(0, 0), (1, 1)]


def test_kalman_prediction_and_ego_identity_transform() -> None:
    tracker = CustomStreetlightTracker(TrackerConfig(mode="centre_scale_kalman"))
    tracker.update([detection([10, 10, 4, 4])])
    tracker.update([detection([12, 10, 4, 4])])
    predicted = tracker._tracks[0].predicted_centre
    assert np.isfinite(predicted).all()
    assert 10.0 < predicted[0] < 12.0
    transformed = transform_points(np.array([[1.0, 2.0]]), np.eye(3))
    assert transformed.tolist() == [[1.0, 2.0]]
    moved = transform_points(np.array([[1.0, 2.0]]), np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]))
    assert moved.tolist() == [[6.0, 0.0]]
