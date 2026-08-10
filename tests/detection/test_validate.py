from types import SimpleNamespace

from rbccps_od.training.validate import validation_summary


def test_validation_summary_extracts_yolo_box_metrics():
    result = SimpleNamespace(
        box=SimpleNamespace(mp=0.8, mr=0.6, map50=0.7, map=0.4),
        speed={"inference": 3.2},
    )

    summary = validation_summary(result, {"split": "test"})

    assert summary["metrics"]["precision"] == 0.8
    assert summary["metrics"]["recall"] == 0.6
    assert summary["metrics"]["f1"] == 0.685714
    assert summary["metrics"]["map50"] == 0.7
    assert summary["metrics"]["map50_95"] == 0.4
    assert summary["speed"]["inference"] == 3.2
