from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


SEED = 20260515


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a mixed local-night + Open Images training dataset.")
    parser.add_argument("--local-root", required=True, help="Local YOLO dataset root, e.g. annotation_automation_v3/yolo_dataset.")
    parser.add_argument("--external-root", required=True, help="External Open Images YOLO dataset root.")
    parser.add_argument("--output-root", required=True, help="Destination mixed dataset root.")
    parser.add_argument("--external-total-ratio", type=float, default=0.15, help="External train image count as a fraction of local train image count.")
    parser.add_argument("--external-positive-fraction", type=float, default=0.8, help="Fraction of sampled external train images that should be positives.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_split_items(dataset_root: Path, split: str) -> list[dict[str, object]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    rows: list[dict[str, object]] = []
    for image_path in sorted(image_dir.glob("*")):
        label_path = label_dir / f"{image_path.stem}.txt"
        label_text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        is_positive = any(line.strip() for line in label_text.splitlines())
        rows.append({"image_path": image_path, "label_path": label_path, "is_positive": is_positive})
    return rows


def sample_external_train(items: list[dict[str, object]], total_target: int, positive_fraction: float) -> list[dict[str, object]]:
    positives = [item for item in items if item["is_positive"]]
    negatives = [item for item in items if not item["is_positive"]]
    rng = random.Random(SEED)

    pos_target = min(len(positives), round(total_target * positive_fraction))
    neg_target = min(len(negatives), max(0, total_target - pos_target))
    if pos_target + neg_target < total_target:
        pos_target = min(len(positives), total_target - neg_target)

    sampled = rng.sample(positives, pos_target) + rng.sample(negatives, neg_target)
    rng.shuffle(sampled)
    return sampled


def copy_item(image_path: Path, label_path: Path, out_image: Path, out_label: Path) -> None:
    ensure_dir(out_image.parent)
    ensure_dir(out_label.parent)
    shutil.copy2(image_path, out_image)
    shutil.copy2(label_path, out_label)


def main() -> None:
    args = parse_args()
    local_root = Path(args.local_root).resolve()
    external_root = Path(args.external_root).resolve()
    output_root = Path(args.output_root).resolve()

    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "valid", "test"):
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    local_train = load_split_items(local_root, "train")
    local_valid = load_split_items(local_root, "valid")
    local_test = load_split_items(local_root, "test")
    external_train = load_split_items(external_root, "train")

    external_target = max(1, round(len(local_train) * args.external_total_ratio))
    sampled_external = sample_external_train(external_train, external_target, args.external_positive_fraction)

    manifest_rows: list[dict[str, str]] = []

    for split, items, source_tag in (
        ("train", local_train, "local"),
        ("valid", local_valid, "local"),
        ("test", local_test, "local"),
    ):
        for item in items:
            image_path = Path(item["image_path"])
            label_path = Path(item["label_path"])
            out_image = output_root / "images" / split / image_path.name
            out_label = output_root / "labels" / split / label_path.name
            copy_item(image_path, label_path, out_image, out_label)
            manifest_rows.append(
                {
                    "split": split,
                    "source": source_tag,
                    "source_image": str(image_path),
                    "output_image": str(out_image),
                    "is_positive": "1" if item["is_positive"] else "0",
                }
            )

    for item in sampled_external:
        image_path = Path(item["image_path"])
        label_path = Path(item["label_path"])
        out_image = output_root / "images" / "train" / f"ext__{image_path.name}"
        out_label = output_root / "labels" / "train" / f"ext__{label_path.name}"
        copy_item(image_path, label_path, out_image, out_label)
        manifest_rows.append(
            {
                "split": "train",
                "source": "openimages_external_train",
                "source_image": str(image_path),
                "output_image": str(out_image),
                "is_positive": "1" if item["is_positive"] else "0",
            }
        )

    dataset_yaml = "\n".join(
        [
            f"path: {output_root.as_posix()}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            "",
            "names:",
            "  0: streetlight",
            "",
        ]
    )
    (output_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    ensure_dir(output_root / "manifests")
    with (output_root / "manifests" / "mixed_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "source", "source_image", "output_image", "is_positive"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_root / "manifests" / "mixed_summary.json").write_text(
        json.dumps(
            {
                "local_train_images": len(local_train),
                "local_valid_images": len(local_valid),
                "local_test_images": len(local_test),
                "external_train_sampled": len(sampled_external),
                "external_total_ratio": args.external_total_ratio,
                "external_positive_fraction": args.external_positive_fraction,
                "external_eval_root": str(external_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "external_train_sampled": len(sampled_external)}, indent=2))


if __name__ == "__main__":
    main()
