from argparse import Namespace
from pathlib import Path

from rbccps_od.training.ablation import (
    CASE_BY_NAME,
    EXPERIMENT_BY_NAME,
    _artifact_name_for,
    _config_for,
    build_wandb_sweep_config,
    selected_cases,
    selected_experiments,
)
from rbccps_od.training.yolo26m_finetune import training_kwargs


def test_selected_experiments_defaults_to_all():
    assert [experiment.name for experiment in selected_experiments(["all"])] == [
        "original",
        "zerodce_enhanced",
        "zero_dce_retinex_reflectance",
        "retinex_decomposition",
    ]


def test_selected_experiments_accepts_aliases_and_typo():
    assert [experiment.name for experiment in selected_experiments(["zerodce", "retinex-deocmposition"])] == [
        "zerodce_enhanced",
        "retinex_decomposition",
    ]


def test_selected_cases_defaults_to_stage1_grid():
    assert [case.name for case in selected_cases(None)] == [
        "baseline",
        "geometry",
        "cse",
        "geometry_cse",
    ]


def test_selected_cases_can_include_negative_stage():
    assert [case.name for case in selected_cases(["stage2"])] == [
        "negative",
        "negative_cse",
        "negative_geometry",
        "all_modules",
    ]


def test_config_for_adds_wandb_ablation_metadata(tmp_path: Path):
    args = Namespace(
        imgsz=1280,
        epochs=60,
        batch=4,
        device="0",
        project=str(tmp_path / "runs"),
        patience=20,
        workers=8,
        cache=False,
        close_mosaic=10,
        exist_ok=False,
        wandb=True,
        wandb_mode="offline",
        wandb_project="streetlight-tests",
        wandb_entity=None,
        wandb_group="ablation-tests",
        wandb_tags=["streetlight", "yolo26m"],
        no_wandb_artifacts=False,
        seed=123,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        negative_mask_root=None,
    )

    config = _config_for(
        EXPERIMENT_BY_NAME["zerodce_enhanced"],
        tmp_path / "dataset.yaml",
        args,
        "yolo26m.pt",
    )

    assert config.wandb is not None
    assert config.wandb.enabled is True
    assert config.wandb.project == "streetlight-tests"
    assert "zerodce_enhanced" in config.wandb.tags
    assert config.wandb.config is not None
    assert config.wandb.config["ablation_experiment"] == "zerodce_enhanced"
    assert config.wandb.config["ablation_case"] == "baseline"
    assert config.wandb.config["model_parameters"] == 20_000_000


def test_config_for_adds_module_flags_to_run_and_artifact(tmp_path: Path):
    args = Namespace(
        imgsz=640,
        epochs=5,
        batch=2,
        device="cpu",
        project=str(tmp_path / "runs"),
        patience=2,
        workers=0,
        cache=False,
        close_mosaic=0,
        exist_ok=True,
        wandb=False,
        wandb_mode="disabled",
        wandb_project="streetlight-tests",
        wandb_entity=None,
        wandb_group="ablation-tests",
        wandb_tags=[],
        no_wandb_artifacts=True,
        seed=42,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        negative_mask_root=str(tmp_path / "masks"),
        negative_mask_loss_weight=0.5,
        allow_mask_unsafe_augmentations=False,
    )

    experiment = EXPERIMENT_BY_NAME["original"]
    case = CASE_BY_NAME["all_modules"]
    config = _config_for(experiment, tmp_path / "dataset.yaml", args, "yolo26m.pt", case)

    assert config.name == "streetlight_yolo26m_negative_geometry_cse_original"
    assert _artifact_name_for(experiment, case) == "original__negative-geometry-cse"
    assert config.use_geometry_attention is True
    assert config.use_cse is True
    assert config.use_negative_attention is True
    assert config.negative_mask_root == (tmp_path / "masks").resolve()
    assert config.negative_mask_loss_weight == 0.5
    assert config.mask_safe_augmentations is True

    kwargs = training_kwargs(config)
    assert kwargs["mosaic"] == 0.0
    assert kwargs["fliplr"] == 0.0


def test_wandb_sweep_config_covers_report_grid():
    sweep = build_wandb_sweep_config(["original"])

    assert sweep["method"] == "grid"
    parameters = sweep["parameters"]
    assert parameters["use_geometry"]["values"] == [False, True]
    assert parameters["use_cse"]["values"] == [False, True]
    assert parameters["use_negative"]["values"] == [False, True]
    assert parameters["data_variant"]["values"] == ["original"]
