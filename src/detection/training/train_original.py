from __future__ import annotations

import argparse
from pathlib import Path

from rbccps_od.training.original_dataset import (
    PreparedOriginalDataset,
    prepare_original_yolo_dataset,
)
from rbccps_od.training.yolo26m_finetune import (
    FineTuneConfig,
    WandbConfig,
    resolve_repo_path,
    resolve_yolo26m_model,
    run_yolo26m_finetune,
    save_trained_weights,
    training_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO26m on the original processed streetlight images."
    )
    parser.add_argument(
        "--image-dir",
        default="datasets/processed/original/images",
        help="Flat source image directory to train from.",
    )
    parser.add_argument(
        "--label-dir",
        default=None,
        help="Flat source label directory. Defaults to the sibling labels directory.",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/processed/original-yolo26m",
        help="Generated YOLO training view with split images and labels.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Pretrained YOLO26m weights. If omitted, local yolo26m.pt is used "
            "when present; otherwise Ultralytics downloads yolo26m.pt."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs", help="Ultralytics project output directory.")
    parser.add_argument("--name", default="streetlight_yolo26m_original", help="Run name.")
    parser.add_argument(
        "--artifact-dir",
        default="models/fine_tuned/yolo26m_ablation/original",
        help="Stable directory where best.pt, last.pt, and metadata.json are copied after training.",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic augmentation in late epochs.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse an existing run name.")
    parser.add_argument("--wandb", action="store_true", help="Log this fine-tuning run to Weights & Biases.")
    parser.add_argument("--wandb-project", default="rbccps-yolo26m-ablation", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
    parser.add_argument("--wandb-group", default="yolo26m-training-ablation", help="W&B group for this run.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B logging mode. Use offline on restricted networks.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=["streetlight", "yolo26m", "training-ablation", "original"],
        help="Extra W&B tags for this run.",
    )
    parser.add_argument("--no-wandb-artifacts", action="store_true", help="Do not upload best.pt/last.pt to W&B.")
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy", "symlink"),
        default="auto",
        help="How to materialize the generated dataset view.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated dataset-view files.")
    parser.add_argument("--prepare-only", action="store_true", help="Build the YOLO dataset view but do not train.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration without starting training.")
    return parser.parse_args()


def _print_dataset_summary(prepared: PreparedOriginalDataset) -> None:
    print(f"dataset_yaml={prepared.dataset_yaml}")
    print(f"manifest={prepared.manifest}")
    for split in ("train", "valid", "test"):
        stats = prepared.stats[split]
        print(
            f"{split}: images={stats.images}, positives={stats.positives}, "
            f"negatives={stats.negatives}, boxes={stats.boxes}"
        )


def main() -> None:
    args = parse_args()

    image_dir = resolve_repo_path(args.image_dir)
    label_dir = resolve_repo_path(args.label_dir) if args.label_dir else image_dir.parent / "labels"
    dataset_root = resolve_repo_path(args.dataset_root)
    artifact_dir = resolve_repo_path(args.artifact_dir)

    prepared = prepare_original_yolo_dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        output_root=dataset_root,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )

    config = FineTuneConfig(
        model=resolve_yolo26m_model(args.model),
        data=prepared.dataset_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=resolve_repo_path(args.project),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
        exist_ok=args.exist_ok,
        wandb=WandbConfig(
            enabled=args.wandb and args.wandb_mode != "disabled",
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=tuple(args.wandb_tags),
            mode=args.wandb_mode if args.wandb_mode != "disabled" else None,
            log_artifacts=not args.no_wandb_artifacts,
            config={
                "ablation_experiment": "original",
                "ablation_artifact_name": "original",
                "dataset_yaml": str(prepared.dataset_yaml),
                "model_family": "YOLO26m",
                "model_parameters": 20_000_000,
            },
        ),
    )

    _print_dataset_summary(prepared)
    print(f"model={config.model}")
    for key, value in training_kwargs(config).items():
        print(f"{key}={value}")
    print(f"artifact_dir={artifact_dir}")
    if config.wandb and config.wandb.enabled:
        print(f"wandb_project={config.wandb.project}")
        print(f"wandb_group={config.wandb.group}")
        print(f"wandb_tags={','.join(config.wandb.tags)}")
        print(f"wandb_mode={config.wandb.mode}")

    if args.prepare_only or args.dry_run:
        return

    run = run_yolo26m_finetune(config)
    saved = save_trained_weights(run, artifact_dir)
    for key, value in saved.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
