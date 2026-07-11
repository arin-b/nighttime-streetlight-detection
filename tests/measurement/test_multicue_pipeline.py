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
