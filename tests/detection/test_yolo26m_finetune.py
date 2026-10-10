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


def test_uniform_mask_safe_augmentations_match_across_factorial_cases(tmp_path: Path):
    from dataclasses import replace

    from rbccps_od.training.yolo26m_finetune import FineTuneConfig, training_kwargs

    baseline = FineTuneConfig(
        model="yolo26m.pt",
        data=tmp_path / "dataset.yaml",
        imgsz=64,
        epochs=1,
        batch=1,
        device="cpu",
        project=tmp_path / "runs",
        name="baseline",
        patience=1,
        workers=0,
        cache=False,
        close_mosaic=0,
        uniform_mask_safe_augmentations=True,
    )
    negative = replace(
        baseline,
        name="negative",
        use_negative_attention=True,
        negative_mask_root=tmp_path / "masks",
    )
    keys = ("degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
            "mosaic", "mixup", "copy_paste", "cutmix")
    baseline_args = training_kwargs(baseline)
    negative_args = training_kwargs(negative)
    assert all(baseline_args[key] == negative_args[key] == 0.0 for key in keys)
