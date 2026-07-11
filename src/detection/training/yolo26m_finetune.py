from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbccps_od.config.paths import ensure_dir, repo_root
from rbccps_od.models.checkpoint_resolver import resolve_checkpoint
from rbccps_od.training.dataset_yaml import resolve_dataset_yaml_for_runtime


@dataclass(frozen=True)
class FineTuneConfig:
    model: str
    data: Path
    imgsz: int
    epochs: int
    batch: int
    device: str
    project: Path
    name: str
    patience: int
    workers: int
    cache: bool
    close_mosaic: int
    exist_ok: bool = False
    wandb: "WandbConfig | None" = None
    seed: int = 42
    deterministic: bool = True
    optimizer: str = "AdamW"
    lr0: float = 1e-3
    weight_decay: float = 5e-4
    use_geometry_attention: bool = False
    use_cse: bool = False
    use_negative_attention: bool = False
    negative_mask_root: Path | None = None
    negative_mask_loss_weight: float = 1.0
    mask_safe_augmentations: bool = True


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "rbccps-yolo26m-ablation"
    entity: str | None = None
    group: str | None = "yolo26m-training-ablation"
    tags: tuple[str, ...] = ()
    mode: str | None = None
    notes: str | None = None
    log_artifacts: bool = True
    config: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrainingRunResult:
    run_dir: Path
    weights_dir: Path
    best_weights: Path
    last_weights: Path


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def resolve_yolo26m_model(model: str | None) -> str:
    if model:
        model_path = Path(model).expanduser()
        repo_candidate = resolve_repo_path(model_path)
        if model_path.is_absolute() or repo_candidate.exists() or len(model_path.parts) > 1:
            return str(repo_candidate)
        return model

    checkpoint = resolve_checkpoint("yolov26_base", allow_missing=True)
    if checkpoint is not None:
        return str(checkpoint)

    for candidate in (
        repo_root() / "datasets" / "yolov26_weights" / "yolo26m.pt",
        repo_root() / "weights" / "yolo26m.pt",
        repo_root() / "models" / "yolo26m.pt",
        repo_root() / "yolo26m.pt",
    ):
        if candidate.exists():
            return str(candidate.resolve())

    return "yolo26m.pt"


def training_kwargs(config: FineTuneConfig) -> dict[str, Any]:
    data_path = config.data.resolve()
    data_arg = data_path
    if data_path.suffix.lower() in {".yaml", ".yml"} and data_path.exists():
        data_arg = resolve_dataset_yaml_for_runtime(data_path)

    kwargs = {
        "data": str(data_arg),
        "imgsz": config.imgsz,
        "epochs": config.epochs,
        "batch": config.batch,
        "device": config.device,
        "project": str(config.project.resolve()),
        "name": config.name,
        "patience": config.patience,
        "workers": config.workers,
        "cache": config.cache,
        "close_mosaic": config.close_mosaic,
        "exist_ok": config.exist_ok,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "weight_decay": config.weight_decay,
    }
    if config.use_negative_attention and config.negative_mask_root is not None and config.mask_safe_augmentations:
        kwargs.update(
            {
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
            }
        )
    return kwargs


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _safe_wandb_config(trainer: Any, config: FineTuneConfig) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "data": config.data,
        "imgsz": config.imgsz,
        "epochs": config.epochs,
        "batch": config.batch,
        "device": config.device,
        "run_name": config.name,
        "seed": config.seed,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "weight_decay": config.weight_decay,
        "use_geometry_attention": config.use_geometry_attention,
        "use_cse": config.use_cse,
        "use_negative_attention": config.use_negative_attention,
        "negative_mask_root": config.negative_mask_root,
        "negative_mask_loss_weight": config.negative_mask_loss_weight,
        "mask_safe_augmentations": config.mask_safe_augmentations,
        "ultralytics_args": vars(trainer.args),
    }
    if config.wandb and config.wandb.config:
        payload.update(config.wandb.config)
    return _jsonable(payload)


def _log_wandb_plots(wandb: Any, trainer: Any, step: int, seen: set[Path]) -> None:
    plots = getattr(trainer, "plots", {}) or {}
    validator = getattr(trainer, "validator", None)
    if validator is not None:
        plots = plots | (getattr(validator, "plots", {}) or {})
    for path in plots:
        plot_path = Path(path)
        if plot_path in seen or not plot_path.exists():
            continue
        wandb.log({plot_path.stem: wandb.Image(str(plot_path))}, step=step)
        seen.add(plot_path)


def attach_wandb_callbacks(yolo: Any, config: FineTuneConfig) -> None:
    wandb_config = config.wandb
    if not wandb_config or not wandb_config.enabled:
        return

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "W&B logging was requested, but wandb is not installed. "
            "Install it with `./streetlight-env/bin/python -m pip install wandb`."
        ) from exc

    run_holder: dict[str, Any] = {}
    logged_plots: set[Path] = set()

    def on_pretrain_routine_start(trainer: Any) -> None:
        if wandb.run is not None:
            run_holder["run"] = wandb.run
            return
        init_kwargs: dict[str, Any] = {
            "project": wandb_config.project,
            "name": config.name,
            "group": wandb_config.group,
            "tags": list(wandb_config.tags),
            "notes": wandb_config.notes,
            "dir": str(trainer.save_dir),
            "config": _safe_wandb_config(trainer, config),
        }
        if wandb_config.entity:
            init_kwargs["entity"] = wandb_config.entity
        if wandb_config.mode:
            init_kwargs["mode"] = wandb_config.mode
        run_holder["run"] = wandb.init(**init_kwargs)

    def on_train_epoch_end(trainer: Any) -> None:
        if wandb.run is None:
            return
        step = int(getattr(trainer, "epoch", 0)) + 1
        metrics: dict[str, Any] = {}
        if getattr(trainer, "tloss", None) is not None:
            metrics.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
        metrics.update(getattr(trainer, "lr", {}) or {})
        if metrics:
            wandb.log(metrics, step=step)

    def on_fit_epoch_end(trainer: Any) -> None:
        if wandb.run is None:
            return
        step = int(getattr(trainer, "epoch", 0)) + 1
        metrics = getattr(trainer, "metrics", {}) or {}
        if metrics:
            wandb.log(metrics, step=step)
        _log_wandb_plots(wandb, trainer, step, logged_plots)

    def on_train_end(trainer: Any) -> None:
        if wandb.run is None:
            return
        step = int(getattr(trainer, "epoch", 0)) + 1
        _log_wandb_plots(wandb, trainer, step, logged_plots)
        if wandb_config.log_artifacts:
            artifact = wandb.Artifact(name=f"{config.name}_weights", type="model")
            added = False
            for path_attr in ("best", "last"):
                value = getattr(trainer, path_attr, None)
                if not value:
                    continue
                weights = Path(value)
                if weights.exists():
                    artifact.add_file(str(weights), name=f"{path_attr}.pt")
                    added = True
            if added:
                wandb.log_artifact(artifact, aliases=["latest"])
        wandb.finish()

    yolo.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
    yolo.add_callback("on_train_epoch_end", on_train_epoch_end)
    yolo.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    yolo.add_callback("on_train_end", on_train_end)


def _run_dir_from(result: Any, yolo: Any, config: FineTuneConfig) -> Path:
    for source in (result, getattr(yolo, "trainer", None)):
        save_dir = getattr(source, "save_dir", None)
        if save_dir:
            return Path(save_dir).resolve()
    return (config.project / config.name).resolve()


def run_yolo26m_finetune(config: FineTuneConfig) -> TrainingRunResult:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ensure_dir(repo_root() / "_ultralytics_config")))
    _set_reproducible_seed(config.seed)

    yolo = _build_yolo_for_config(config)
    attach_wandb_callbacks(yolo, config)
    trainer = _trainer_for_config(config)
    train_args = training_kwargs(config)
    if trainer is not None:
        result = yolo.train(trainer=trainer, **train_args)
    else:
        result = yolo.train(**train_args)
    run_dir = _run_dir_from(result, yolo, config)
    weights_dir = run_dir / "weights"
    return TrainingRunResult(
        run_dir=run_dir,
        weights_dir=weights_dir,
        best_weights=weights_dir / "best.pt",
        last_weights=weights_dir / "last.pt",
    )


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _build_yolo_for_config(config: FineTuneConfig) -> Any:
    if config.use_geometry_attention or config.use_cse or config.use_negative_attention:
        from rbccps_od.models.yolo_ablation import build_yolo26_ablation_model

        return build_yolo26_ablation_model(
            config.model,
            use_geometry_attention=config.use_geometry_attention,
            use_cse=config.use_cse,
            use_negative_attention=config.use_negative_attention,
            negative_mask_loss_weight=config.negative_mask_loss_weight,
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install the training extras before running this command.") from exc
    return YOLO(config.model)


def _trainer_for_config(config: FineTuneConfig) -> Any | None:
    if not config.use_negative_attention:
        return None

    from rbccps_od.models.yolo_ablation import negative_mask_trainer

    return negative_mask_trainer(config.negative_mask_root, loss_weight=config.negative_mask_loss_weight)


def save_trained_weights(run: TrainingRunResult, artifact_dir: Path) -> dict[str, str]:
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {
        "run_dir": str(run.run_dir),
        "weights_dir": str(run.weights_dir),
    }
    for label, source in (("best", run.best_weights), ("last", run.last_weights)):
        if not source.exists():
            raise FileNotFoundError(f"Expected trained {label} weights at {source}")
        target = artifact_dir / source.name
        shutil.copy2(source, target)
        saved[f"{label}_weights"] = str(target)

    for artifact_name in ("results.csv", "args.yaml", "confusion_matrix.png", "results.png"):
        source = run.run_dir / artifact_name
        if source.exists():
            target = artifact_dir / artifact_name
            shutil.copy2(source, target)
            saved[artifact_name.replace(".", "_")] = str(target)

    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    saved["metadata"] = str(metadata_path)
    return saved
