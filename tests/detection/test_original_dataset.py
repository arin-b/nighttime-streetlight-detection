from pathlib import Path

import yaml

from rbccps_od.training.original_dataset import prepare_original_yolo_dataset


def test_original_dataset_yaml_matches_one_class_labels_and_split_paths(tmp_path: Path):
    image_dir = tmp_path / "source" / "images"
    label_dir = tmp_path / "source" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        stem = f"clip_images_{split}_001"
        (image_dir / f"{stem}.jpg").write_bytes(b"image")
        (label_dir / f"{stem.replace('_images_', '_labels_')}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    prepared = prepare_original_yolo_dataset(image_dir, label_dir, tmp_path / "prepared")
    generated = yaml.safe_load(prepared.dataset_yaml.read_text(encoding="utf-8"))
    static = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "src/detection/config/original.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert generated["nc"] == 1
    assert generated["names"] == {0: "streetlight"}
    assert {key: generated[key] for key in ("train", "val", "test", "nc", "names")} == {
        key: static[key] for key in ("train", "val", "test", "nc", "names")
    }
    for split in ("train", "valid", "test"):
        yaml_key = "val" if split == "valid" else split
        assert (prepared.dataset_root / generated[yaml_key]).is_dir()
        label = prepared.dataset_root / "labels" / split / f"clip_images_{split}_001.txt"
        assert {int(row.split()[0]) for row in label.read_text(encoding="utf-8").splitlines()} == {0}
