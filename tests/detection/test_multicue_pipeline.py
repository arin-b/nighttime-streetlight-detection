import numpy as np

from audit_pipeline.config import MultiCueConfig
from audit_pipeline.multicue_filter import filter_frame_detections
from rbccps_od.config.schemas import CueWeights
from rbccps_od.domain.tracks import Track
from rbccps_od.pipeline.multicue_stage import MultiCueFilterStage


def test_multicue_pipeline_scores_and_thresholds_tracks():
    track = Track(track_id="t1", bbox=[10.0, 20.0, 5.0, 12.0], score=0.9, history=[[10.0, 20.0, 5.0, 12.0]] * 4)
    stage = MultiCueFilterStage(weights=CueWeights(), threshold=0.6, enabled=True)
    results = stage.run([track], frame_height=100.0)
    assert len(results) == 1
    assert results[0]["aggregate_score"] >= 0.6
    assert results[0]["accepted"] is True
    assert len(results[0]["cues"]) == 4


def test_audit_multicue_filter_uses_weighted_scoring():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{
        "track_id": "t1",
        "xyxy": [10.0, 20.0, 40.0, 60.0],
        "confidence": 0.9,
        "class_id": 0,
        "class_label": "streetlight",
    }]

    results = filter_frame_detections(
        frame,
        detections,
        1,
        MultiCueConfig(aggregation_threshold=0.5),
        track_histories={},
    )

    assert len(results) == 1
    assert results[0].aggregate_score >= 0.5
    assert results[0].kept is True
