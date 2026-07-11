from pathlib import Path

from rbccps_od.training.dataset_yaml import resolve_dataset_yaml_for_runtime


def test_windows_style_dataset_path_is_rewritten(tmp_path: Path):
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join([
            "path: F:/RBCCPS_Directory/datasets/derived/annotation_automation_v3/yolo_dataset",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            "names:",
            "  0: streetlight",
        ]),
        encoding="utf-8",
    )
    resolved = resolve_dataset_yaml_for_runtime(dataset_yaml)
    assert resolved.exists()
    assert str(tmp_path).replace("\\", "/") in resolved.read_text(encoding="utf-8")
