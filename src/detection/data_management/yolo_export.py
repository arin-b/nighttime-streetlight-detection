from __future__ import annotations

from pathlib import Path

from rbccps_od.data_management.corpus_v3 import copy_image, output_dataset_yaml

__all__ = ["copy_image", "output_dataset_yaml", "yolo_line", "dataset_yaml_payload"]


def yolo_line(bbox: list[float], width: int, height: int) -> str:
    x, y, w, h = bbox
    x_c = (x + (w / 2.0)) / width
    y_c = (y + (h / 2.0)) / height
    return f"0 {x_c:.6f} {y_c:.6f} {w / width:.6f} {h / height:.6f}"


def dataset_yaml_payload(dataset_root: Path) -> str:
    return "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            "",
            "names:",
            "  0: streetlight",
            "",
        ]
    )
