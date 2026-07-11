from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}

SPLIT_TOKENS = {
    "_images_train_": "train",
    "_images_valid_": "valid",
    "_images_test_": "test",
}


@dataclass
class SplitStats:
    images: int = 0
    positives: int = 0
    negatives: int = 0
    boxes: int = 0


@dataclass
class PreparedOriginalDataset:
    dataset_root: Path
    dataset_yaml: Path
    manifest: Path
    stats: dict[str, SplitStats] = field(default_factory=dict)

    @property
    def image_count(self) -> int:
        return sum(split.images for split in self.stats.values())


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def infer_split(image_path: Path) -> str:
    for token, split in SPLIT_TOKENS.items():
        if token in image_path.stem:
            return split
    raise ValueError(f"Could not infer split from filename: {image_path.name}")


def matching_label(image_path: Path, label_dir: Path) -> Path:
    renamed_stem = image_path.stem.replace("_images_", "_labels_")
    candidates = [
        label_dir / f"{renamed_stem}.txt",
        label_dir / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No matching label found for {image_path.name}")


def _count_label_boxes(label_path: Path) -> int:
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _materialize_file(src: Path, dst: Path, *, mode: str, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "existing"
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlinked"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlinked"
    if mode == "auto":
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied"
    raise ValueError(f"Unknown link mode: {mode}")


def write_dataset_yaml(output_root: Path) -> Path:
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/valid",
                "test: images/test",
                "nc: 2",
                "names:",
                "  0: lamp",
                "  1: pole",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def prepare_original_yolo_dataset(
    image_dir: Path,
    label_dir: Path,
    output_root: Path,
    *,
    link_mode: str = "auto",
    overwrite: bool = False,
) -> PreparedOriginalDataset:
    image_dir = image_dir.resolve()
    label_dir = label_dir.resolve()
    output_root = output_root.resolve()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    images = collect_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    stats = {split: SplitStats() for split in ("train", "valid", "test")}
    rows: list[dict[str, str]] = []

    for image_path in images:
        split = infer_split(image_path)
        label_path = matching_label(image_path, label_dir)
        box_count = _count_label_boxes(label_path)

        output_image = output_root / "images" / split / image_path.name
        output_label = output_root / "labels" / split / f"{image_path.stem}.txt"

        image_action = _materialize_file(
            image_path,
            output_image,
            mode=link_mode,
            overwrite=overwrite,
        )
        label_action = _materialize_file(
            label_path,
            output_label,
            mode=link_mode,
            overwrite=overwrite,
        )

        split_stats = stats[split]
        split_stats.images += 1
        split_stats.boxes += box_count
        if box_count:
            split_stats.positives += 1
        else:
            split_stats.negatives += 1

        rows.append(
            {
                "split": split,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "training_image": str(output_image),
                "training_label": str(output_label),
                "boxes": str(box_count),
                "image_action": image_action,
                "label_action": label_action,
            }
        )

    dataset_yaml = write_dataset_yaml(output_root)
    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return PreparedOriginalDataset(
        dataset_root=output_root,
        dataset_yaml=dataset_yaml,
        manifest=manifest,
        stats=stats,
    )
