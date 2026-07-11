from __future__ import annotations

from pathlib import Path


def resolve_frame_uri(dataset_root: str | Path, image_uri: str) -> Path:
    path = Path(image_uri)
    return path if path.is_absolute() else Path(dataset_root) / path
