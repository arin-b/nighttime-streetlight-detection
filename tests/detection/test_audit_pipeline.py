from types import SimpleNamespace

import numpy as np
import pytest

from audit_pipeline.config import DetectorConfig
from audit_pipeline.detector import resolve_target_classes
from audit_pipeline.measurement import LampMeasurement
from audit_pipeline.run_audit import (
    draw_streetlight_annotations,
    extract_detections,
    keep_target_class_detections,
)


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


def fake_model(names):
    return SimpleNamespace(names=names)


def fake_result():
    return SimpleNamespace(
        names={0: "person", 1: "streetlight"},
        boxes=SimpleNamespace(
            xyxy=FakeTensor([[1, 2, 10, 20], [30, 40, 60, 80]]),
            conf=FakeTensor([0.9, 0.8]),
            id=FakeTensor([11, 12]),
            cls=FakeTensor([0, 1]),
        ),
    )


def test_resolve_target_classes_rejects_coco_default_person():
    cfg = DetectorConfig()

    with pytest.raises(ValueError, match="COCO-style"):
        resolve_target_classes(
            fake_model({0: "person", 2: "car", 9: "traffic light"}),
            cfg,
        )


def test_resolve_target_classes_finds_streetlight_name():
    cfg = DetectorConfig()

    resolved = resolve_target_classes(fake_model({0: "person", 4: "street_light"}), cfg)

    assert resolved == [4]
    assert cfg.target_classes == [4]
    assert cfg.resolved_class_names == {4: "street_light"}


def test_resolve_target_classes_allows_one_class_numeric_alias():
    cfg = DetectorConfig()

    resolved = resolve_target_classes(fake_model({0: "0"}), cfg)

    assert resolved == [0]
    assert cfg.resolved_class_names == {0: "0"}


def test_resolve_target_classes_rejects_explicit_person_id():
    cfg = DetectorConfig(target_classes=[0])

    with pytest.raises(ValueError, match="person"):
        resolve_target_classes(fake_model({0: "person", 1: "streetlight"}), cfg)


def test_extract_and_keep_target_class_detections_suppresses_non_targets():
    detections = extract_detections(fake_result())

    kept, suppressed = keep_target_class_detections(detections, {1})

    assert suppressed == 1
    assert [det["track_id"] for det in kept] == ["track_12"]
    assert kept[0]["class_label"] == "streetlight"


def test_draw_streetlight_annotations_uses_active_detections_only():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = extract_detections(fake_result())
    kept, _ = keep_target_class_detections(detections, {1})
    measurement = LampMeasurement(
        track_id="track_12",
        frame_index=1,
        xyxy=kept[0]["xyxy"],
        mean_brightness=120.0,
        is_on=True,
        detection_confidence=0.8,
    )

    annotated = draw_streetlight_annotations(frame, kept, {"track_12": measurement})

    assert np.count_nonzero(annotated) > 0
    assert np.count_nonzero(annotated[:25, :25]) == 0
