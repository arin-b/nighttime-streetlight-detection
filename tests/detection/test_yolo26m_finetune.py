from pathlib import Path

from rbccps_od.training.yolo26m_finetune import TrainingRunResult, save_trained_weights


def test_save_trained_weights_copies_best_last_and_metadata(tmp_path: Path):
    run_dir = tmp_path / "runs" / "experiment"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"best")
    (weights_dir / "last.pt").write_bytes(b"last")

    saved = save_trained_weights(
        TrainingRunResult(
            run_dir=run_dir,
            weights_dir=weights_dir,
            best_weights=weights_dir / "best.pt",
            last_weights=weights_dir / "last.pt",
        ),
        tmp_path / "artifacts" / "experiment",
    )

    assert Path(saved["best_weights"]).read_bytes() == b"best"
    assert Path(saved["last_weights"]).read_bytes() == b"last"
    assert Path(saved["metadata"]).exists()
