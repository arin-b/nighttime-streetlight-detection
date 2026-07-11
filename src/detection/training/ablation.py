from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rbccps_od.training.original_dataset import PreparedOriginalDataset, prepare_original_yolo_dataset
from rbccps_od.training.yolo26m_finetune import (
    FineTuneConfig,
    WandbConfig,
    resolve_repo_path,
    resolve_yolo26m_model,
    run_yolo26m_finetune,
    save_trained_weights,
    training_kwargs,
)


@dataclass(frozen=True)
class AblationExperiment:
    name: str
    run_name: str
    artifact_name: str
    dataset_yaml: str | None = None
    uses_original_view: bool = False


@dataclass(frozen=True)
class AblationCase:
    name: str
    run_label: str
    artifact_label: str
    use_geometry: bool = False
    use_cse: bool = False
    use_negative: bool = False


EXPERIMENTS: tuple[AblationExperiment, ...] = (
    AblationExperiment(
        name="original",
        run_name="streetlight_yolo26m_original",
        artifact_name="original",
        uses_original_view=True,
    ),
    AblationExperiment(
        name="zerodce_enhanced",
        run_name="streetlight_yolo26m_zerodce_enhanced",
        artifact_name="zerodce-enhanced",
        dataset_yaml="datasets/processed/zerodce-enhanced/dataset.yaml",
    ),
    AblationExperiment(
        name="zero_dce_retinex_reflectance",
        run_name="streetlight_yolo26m_zero_dce_retinex_reflectance",
        artifact_name="zero-dce-retinex-reflectance",
        dataset_yaml="datasets/processed/zero_dce_retinex_reflectance/dataset.yaml",
    ),
    AblationExperiment(
        name="retinex_decomposition",
        run_name="streetlight_yolo26m_retinex_decomposition",
        artifact_name="retinex-decomposition",
        dataset_yaml="datasets/processed/retinex-decomposition/dataset.yaml",
    ),
)

EXPERIMENT_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}
EXPERIMENT_ALIASES = {
    "original": "original",
    "baseline": "original",
    "zerodce": "zerodce_enhanced",
    "zero-dce": "zerodce_enhanced",
    "zerodce-enhanced": "zerodce_enhanced",
    "zerodce_enhanced": "zerodce_enhanced",
    "zero-dce-enhanced": "zerodce_enhanced",
    "zero_dce_enhanced": "zerodce_enhanced",
    "zero-dce-retinex": "zero_dce_retinex_reflectance",
    "zero_dce_retinex": "zero_dce_retinex_reflectance",
    "zero-dce-retinex-reflectance": "zero_dce_retinex_reflectance",
    "zero_dce_retinex_reflectance": "zero_dce_retinex_reflectance",
    "retinex": "retinex_decomposition",
    "retinex-decomposition": "retinex_decomposition",
    "retinex_decomposition": "retinex_decomposition",
    "retinex-deocmposition": "retinex_decomposition",
}

BASELINE_CASE = AblationCase(name="baseline", run_label="baseline", artifact_label="baseline")
STAGE1_CASES: tuple[AblationCase, ...] = (
    BASELINE_CASE,
    AblationCase(name="geometry", run_label="geometry", artifact_label="geometry", use_geometry=True),
    AblationCase(name="cse", run_label="cse", artifact_label="cse", use_cse=True),
    AblationCase(
        name="geometry_cse",
        run_label="geometry_cse",
        artifact_label="geometry-cse",
        use_geometry=True,
        use_cse=True,
    ),
)
NEGATIVE_CASES: tuple[AblationCase, ...] = (
    AblationCase(name="negative", run_label="negative", artifact_label="negative", use_negative=True),
    AblationCase(
        name="negative_cse",
        run_label="negative_cse",
        artifact_label="negative-cse",
        use_negative=True,
        use_cse=True,
    ),
    AblationCase(
        name="negative_geometry",
        run_label="negative_geometry",
        artifact_label="negative-geometry",
        use_negative=True,
        use_geometry=True,
    ),
    AblationCase(
        name="all_modules",
        run_label="negative_geometry_cse",
        artifact_label="negative-geometry-cse",
        use_negative=True,
        use_geometry=True,
        use_cse=True,
    ),
)
ABLATION_CASES: tuple[AblationCase, ...] = STAGE1_CASES + NEGATIVE_CASES
CASE_BY_NAME = {case.name: case for case in ABLATION_CASES}
CASE_ALIASES = {
    "baseline": "baseline",
    "base": "baseline",
    "none": "baseline",
    "geometry": "geometry",
    "geo": "geometry",
    "cse": "cse",
    "geometry-cse": "geometry_cse",
    "geometry_cse": "geometry_cse",
    "geo-cse": "geometry_cse",
    "negative": "negative",
    "neg": "negative",
    "negative-cse": "negative_cse",
    "negative_cse": "negative_cse",
    "neg-cse": "negative_cse",
    "negative-geometry": "negative_geometry",
    "negative_geometry": "negative_geometry",
    "neg-geo": "negative_geometry",
    "all-modules": "all_modules",
    "all_modules": "all_modules",
    "all": "all",
    "stage1": "stage1",
    "stage2": "stage2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO26m fine-tuning ablation studies.")
    parser.add_argument(
        "experiments",
        nargs="*",
        default=["all"],
        help="Experiments to run. Use all, original, zerodce, zero-dce-retinex, or retinex.",
    )
    parser.add_argument(
        "--data-variant",
        default=None,
        help="Single data variant for W&B sweep agents. Uses the same aliases as the positional experiments.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help=(
            "Ablation cases to run. Use stage1, stage2, all, baseline, geometry, cse, "
            "geometry-cse, negative, negative-cse, negative-geometry, or all-modules."
        ),
    )
    parser.add_argument("--single-run", action="store_true", help="Run exactly one case from the module flags.")
    parser.add_argument("--use-geometry", action="store_true", help="Enable geometry-aware attention.")
    parser.add_argument("--use-cse", action="store_true", help="Enable channel squeeze-and-excitation blocks.")
    parser.add_argument("--use-negative", action="store_true", help="Enable negative-region feature suppression.")
    parser.add_argument(
        "--model",
        default=None,
        help="Pretrained YOLO26m weights. Defaults to local yolo26m.pt if available, otherwise yolo26m.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed logged and passed to Ultralytics.")
    parser.add_argument("--optimizer", default="AdamW", help="Ultralytics optimizer name.")
    parser.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--project", default="runs/yolo26m_ablation", help="Ultralytics project output directory.")
    parser.add_argument(
        "--artifact-root",
        default="models/fine_tuned/yolo26m_ablation",
        help="Root where each experiment gets its own best.pt, last.pt, and metadata.json.",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic augmentation in late epochs.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse existing run names.")
    parser.add_argument("--wandb", action="store_true", help="Log each ablation run to Weights & Biases.")
    parser.add_argument("--wandb-project", default="rbccps-yolo26m-ablation", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
    parser.add_argument("--wandb-group", default="yolo26m-training-ablation", help="W&B group for these runs.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B logging mode. Use offline on restricted networks.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=["streetlight", "yolo26m", "training-ablation"],
        help="Extra W&B tags applied to every ablation run.",
    )
    parser.add_argument("--no-wandb-artifacts", action="store_true", help="Do not upload best.pt/last.pt to W&B.")
    parser.add_argument(
        "--print-wandb-sweep",
        action="store_true",
        help="Print a W&B grid sweep config for the full report ablation grid and exit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an experiment when its artifact best.pt already exists.",
    )
    parser.add_argument(
        "--original-image-dir",
        default="datasets/processed/original/images",
        help="Flat source image directory for the original baseline ablation.",
    )
    parser.add_argument(
        "--original-label-dir",
        default=None,
        help="Flat source label directory for the original baseline. Defaults to sibling labels.",
    )
    parser.add_argument(
        "--original-dataset-root",
        default="datasets/processed/original-yolo26m",
        help="Generated YOLO training view for the original baseline.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy", "symlink"),
        default="auto",
        help="How to materialize the generated original dataset view.",
    )
    parser.add_argument(
        "--negative-mask-root",
        default=None,
        help=(
            "Root directory containing negative segmentation masks. Filenames may mirror the image split layout "
            "or share image stems with .png/.jpg/.npy/.pt suffixes."
        ),
    )
    parser.add_argument(
        "--negative-mask-loss-weight",
        type=float,
        default=1.0,
        help="Lambda for Ldet + lambda * Lmask in negative-attention ablations.",
    )
    parser.add_argument(
        "--allow-mask-unsafe-augmentations",
        action="store_true",
        help=(
            "Keep YOLO geometric augmentations for negative-mask runs. By default they are disabled because "
            "external mask files cannot receive the same random transforms as the images."
        ),
    )
    parser.add_argument(
        "--allow-dataset-negative-masks",
        action="store_true",
        help="Allow negative cases without --negative-mask-root when a custom dataset already provides masks.",
    )
    parser.add_argument(
        "--overwrite-original-view",
        action="store_true",
        help="Overwrite files in the generated original dataset view.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and print configs without training.")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without training.")
    return parser.parse_args()


def selected_experiments(values: list[str]) -> list[AblationExperiment]:
    if not values or any(value.lower() == "all" for value in values):
        return list(EXPERIMENTS)

    selected: list[AblationExperiment] = []
    seen: set[str] = set()
    for value in values:
        key = EXPERIMENT_ALIASES.get(value.lower())
        if key is None:
            valid = ", ".join(experiment.name for experiment in EXPERIMENTS)
            raise SystemExit(f"Unknown ablation experiment '{value}'. Valid experiments: {valid}")
        if key not in seen:
            selected.append(EXPERIMENT_BY_NAME[key])
            seen.add(key)
    return selected


def selected_cases(values: list[str] | None) -> list[AblationCase]:
    requested = values or ["stage1"]
    if any(value.lower() == "all" for value in requested):
        return list(ABLATION_CASES)

    selected: list[AblationCase] = []
    seen: set[str] = set()
    for value in requested:
        alias = CASE_ALIASES.get(value.lower())
        if alias == "stage1":
            candidates = STAGE1_CASES
        elif alias == "stage2":
            candidates = NEGATIVE_CASES
        elif alias in CASE_BY_NAME:
            candidates = (CASE_BY_NAME[alias],)
        else:
            valid = ", ".join(case.name for case in ABLATION_CASES)
            raise SystemExit(f"Unknown ablation case '{value}'. Valid cases: stage1, stage2, all, {valid}")
        for candidate in candidates:
            if candidate.name not in seen:
                selected.append(candidate)
                seen.add(candidate.name)
    return selected


def case_from_flags(*, use_geometry: bool, use_cse: bool, use_negative: bool) -> AblationCase:
    for case in ABLATION_CASES:
        if (
            case.use_geometry == use_geometry
            and case.use_cse == use_cse
            and case.use_negative == use_negative
        ):
            return case
    return AblationCase(
        name="custom",
        run_label="custom",
        artifact_label="custom",
        use_geometry=use_geometry,
        use_cse=use_cse,
        use_negative=use_negative,
    )


def build_wandb_sweep_config(data_variants: list[str] | None = None) -> dict[str, object]:
    return {
        "method": "grid",
        "parameters": {
            "use_geometry": {"values": [False, True]},
            "use_cse": {"values": [False, True]},
            "use_negative": {"values": [False, True]},
            "data_variant": {"values": data_variants or [experiment.name for experiment in EXPERIMENTS]},
        },
    }


def _print_original_summary(prepared: PreparedOriginalDataset) -> None:
    print(f"dataset_yaml={prepared.dataset_yaml}")
    print(f"manifest={prepared.manifest}")
    for split in ("train", "valid", "test"):
        stats = prepared.stats[split]
        print(
            f"{split}: images={stats.images}, positives={stats.positives}, "
            f"negatives={stats.negatives}, boxes={stats.boxes}"
        )


def _dataset_yaml_for(experiment: AblationExperiment, args: argparse.Namespace) -> Path:
    if not experiment.uses_original_view:
        assert experiment.dataset_yaml is not None
        dataset_yaml = resolve_repo_path(experiment.dataset_yaml)
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found for {experiment.name}: {dataset_yaml}")
        print(f"dataset_yaml={dataset_yaml}")
        return dataset_yaml

    image_dir = resolve_repo_path(args.original_image_dir)
    label_dir = resolve_repo_path(args.original_label_dir) if args.original_label_dir else image_dir.parent / "labels"
    prepared = prepare_original_yolo_dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        output_root=resolve_repo_path(args.original_dataset_root),
        link_mode=args.link_mode,
        overwrite=args.overwrite_original_view,
    )
    _print_original_summary(prepared)
    return prepared.dataset_yaml


def _run_name_for(experiment: AblationExperiment, case: AblationCase) -> str:
    if case.name == "baseline":
        return experiment.run_name
    return f"streetlight_yolo26m_{case.run_label}_{experiment.name}"


def _artifact_name_for(experiment: AblationExperiment, case: AblationCase) -> str:
    if case.name == "baseline":
        return experiment.artifact_name
    return f"{experiment.artifact_name}__{case.artifact_label}"


def _negative_mask_root(args: argparse.Namespace) -> Path | None:
    value = getattr(args, "negative_mask_root", None)
    return resolve_repo_path(value) if value else None


def _config_for(
    experiment: AblationExperiment,
    data: Path,
    args: argparse.Namespace,
    model: str,
    case: AblationCase = BASELINE_CASE,
) -> FineTuneConfig:
    artifact_name = _artifact_name_for(experiment, case)
    base_tags = getattr(args, "wandb_tags", ["streetlight", "yolo26m", "training-ablation"])
    wandb_tags = tuple(dict.fromkeys([*base_tags, experiment.name, experiment.artifact_name, case.name]))
    return FineTuneConfig(
        model=model,
        data=data,
        imgsz=getattr(args, "imgsz", 1280),
        epochs=getattr(args, "epochs", 60),
        batch=getattr(args, "batch", 4),
        device=getattr(args, "device", "0"),
        project=resolve_repo_path(args.project),
        name=_run_name_for(experiment, case),
        patience=getattr(args, "patience", 20),
        workers=getattr(args, "workers", 8),
        cache=getattr(args, "cache", False),
        close_mosaic=getattr(args, "close_mosaic", 10),
        exist_ok=getattr(args, "exist_ok", False),
        wandb=WandbConfig(
            enabled=getattr(args, "wandb", False) and getattr(args, "wandb_mode", "online") != "disabled",
            project=getattr(args, "wandb_project", "rbccps-yolo26m-ablation"),
            entity=getattr(args, "wandb_entity", None),
            group=getattr(args, "wandb_group", "yolo26m-training-ablation"),
            tags=wandb_tags,
            mode=getattr(args, "wandb_mode", "online") if getattr(args, "wandb_mode", "online") != "disabled" else None,
            log_artifacts=not getattr(args, "no_wandb_artifacts", False),
            config={
                "ablation_experiment": experiment.name,
                "ablation_case": case.name,
                "ablation_artifact_name": artifact_name,
                "dataset_yaml": str(data),
                "model_family": "YOLO26m",
                "model_parameters": 20_000_000,
                "use_geometry_attention": case.use_geometry,
                "use_cse": case.use_cse,
                "use_negative_attention": case.use_negative,
                "negative_mask_loss_weight": getattr(args, "negative_mask_loss_weight", 1.0),
                "mask_safe_augmentations": not getattr(args, "allow_mask_unsafe_augmentations", False),
            },
        ),
        seed=getattr(args, "seed", 42),
        optimizer=getattr(args, "optimizer", "AdamW"),
        lr0=getattr(args, "lr0", 1e-3),
        weight_decay=getattr(args, "weight_decay", 5e-4),
        use_geometry_attention=case.use_geometry,
        use_cse=case.use_cse,
        use_negative_attention=case.use_negative,
        negative_mask_root=_negative_mask_root(args),
        negative_mask_loss_weight=getattr(args, "negative_mask_loss_weight", 1.0),
        mask_safe_augmentations=not getattr(args, "allow_mask_unsafe_augmentations", False),
    )


def main() -> None:
    args = parse_args()
    if args.print_wandb_sweep:
        print(json.dumps(build_wandb_sweep_config(), indent=2))
        return

    experiment_values = [args.data_variant] if args.data_variant else args.experiments
    experiments = selected_experiments(experiment_values)
    if args.cases is None and (args.single_run or args.use_geometry or args.use_cse or args.use_negative):
        cases = [case_from_flags(use_geometry=args.use_geometry, use_cse=args.use_cse, use_negative=args.use_negative)]
    else:
        cases = selected_cases(args.cases)
    model = resolve_yolo26m_model(args.model)
    artifact_root = resolve_repo_path(args.artifact_root)
    total_runs = len(experiments) * len(cases)
    run_index = 0

    for experiment in experiments:
        data = _dataset_yaml_for(experiment, args)
        for case in cases:
            run_index += 1
            print(f"\n[{run_index}/{total_runs}] experiment={experiment.name} case={case.name}")
            artifact_dir = artifact_root / _artifact_name_for(experiment, case)
            if args.skip_existing and (artifact_dir / "best.pt").exists():
                print(f"skipped_existing={artifact_dir / 'best.pt'}")
                continue
            if (
                case.use_negative
                and not args.negative_mask_root
                and not args.allow_dataset_negative_masks
                and not args.prepare_only
                and not args.dry_run
            ):
                raise SystemExit(
                    "Negative-mask ablation cases require --negative-mask-root unless "
                    "--allow-dataset-negative-masks is set."
                )

            config = _config_for(experiment, data, args, model, case)

            print(f"model={config.model}")
            print(f"use_geometry_attention={config.use_geometry_attention}")
            print(f"use_cse={config.use_cse}")
            print(f"use_negative_attention={config.use_negative_attention}")
            if config.negative_mask_root:
                print(f"negative_mask_root={config.negative_mask_root}")
                print(f"negative_mask_loss_weight={config.negative_mask_loss_weight}")
                print(f"mask_safe_augmentations={config.mask_safe_augmentations}")
            for key, value in training_kwargs(config).items():
                print(f"{key}={value}")
            print(f"artifact_dir={artifact_dir}")
            if config.wandb and config.wandb.enabled:
                print(f"wandb_project={config.wandb.project}")
                print(f"wandb_group={config.wandb.group}")
                print(f"wandb_tags={','.join(config.wandb.tags)}")
                print(f"wandb_mode={config.wandb.mode}")

            if args.prepare_only or args.dry_run:
                continue

            run = run_yolo26m_finetune(config)
            saved = save_trained_weights(run, artifact_dir)
            for key, value in saved.items():
                print(f"{key}={value}")


if __name__ == "__main__":
    main()
