from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rbccps_od.io.yaml_io import parse_simple_yaml


def resolve_dataset_yaml_for_runtime(data_path: Path) -> Path:
    data_path = data_path.resolve()
    payload = parse_simple_yaml(data_path)
    path_value = payload.get("path", "")
    train_value = payload.get("train", "")
    path_looks_windows = len(path_value) > 2 and path_value[1:3] in {":\\", ":/"}
    train_exists = False
    if path_value and train_value and not path_looks_windows:
        train_exists = (Path(path_value) / train_value).exists()
    if not (path_looks_windows or (path_value and train_value and not train_exists)):
        return data_path

    yaml_lines = data_path.read_text(encoding="utf-8").splitlines()
    inferred_path = str(data_path.parent.resolve())
    temp_payload = []
    for line in yaml_lines:
        if line.startswith("path:"):
            temp_payload.append(f"path: {inferred_path.replace(os.sep, '/')}")
        else:
            temp_payload.append(line)
    temp_file = Path(tempfile.gettempdir()) / f"{data_path.stem}_resolved.yaml"
    temp_file.write_text("\n".join(temp_payload) + "\n", encoding="utf-8")
    return temp_file
