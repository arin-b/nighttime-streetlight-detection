from argparse import Namespace
from pathlib import Path

from rbccps_od.pipeline.video_runner import _track_summary_rows, write_botsort_config


def test_write_botsort_config_enables_gmc(tmp_path: Path):
    args = Namespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        gmc_method="sparseOptFlow",
        with_reid=False,
    )

    tracker_path = write_botsort_config(args, tmp_path)

    payload = tracker_path.read_text(encoding="utf-8")
    assert "tracker_type: botsort" in payload
    assert "gmc_method: sparseOptFlow" in payload


def test_track_summary_rows_aggregates_track_stats():
    rows = _track_summary_rows(
        {
            "track_1": {
                "track_id": "track_1",
                "first_frame": 1,
                "first_time_sec": 0.0,
                "last_frame": 3,
                "last_time_sec": 0.2,
                "visible_frames": 3,
                "accepted_frames": 2,
                "scores": [0.5, 0.7, 0.9],
                "aggregate_scores": [0.4, 0.8, 0.6],
                "max_history_len": 3,
            }
        }
    )

    assert rows[0]["track_id"] == "track_1"
    assert rows[0]["accepted_any"] is True
    assert rows[0]["mean_score"] == "0.700000"
    assert rows[0]["max_aggregate_score"] == "0.800000"
