from __future__ import annotations

from pathlib import Path


def require_video_path(path: str | Path) -> Path:
    video = Path(path)
    if not video.exists():
        raise FileNotFoundError(f"video path does not exist: {video}")
    return video
