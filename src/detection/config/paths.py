from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    env_root = os.environ.get("RBCCPS_OD_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "scripts" / "annotation_automation").exists():
            return candidate
    return current.parents[3]


def datasets_root() -> Path:
    return repo_root() / "datasets"


def models_root() -> Path:
    return repo_root() / "models"


def model_cache_root() -> Path:
    return repo_root() / ".cache" / "rbccps_od" / "models"


def documentation_root() -> Path:
    return repo_root() / "documentation"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
