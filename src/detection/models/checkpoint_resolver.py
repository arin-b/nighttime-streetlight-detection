from __future__ import annotations

from pathlib import Path

from rbccps_od.config.paths import repo_root
from rbccps_od.models.downloader import cached_asset_path, download_asset
from rbccps_od.models.registry import get_asset


def resolve_checkpoint(name: str, allow_missing: bool = False) -> Path | None:
    spec = get_asset(name)
    if spec.local_path:
        local_target = (repo_root() / spec.local_path).resolve()
        if local_target.exists():
            return local_target
    target = cached_asset_path(name)
    if target.exists():
        return target
    try:
        materialized = download_asset(name)
        if materialized.exists():
            return materialized
    except Exception:
        if allow_missing:
            return None
        raise
    if allow_missing:
        return None
    raise FileNotFoundError(
        f"Checkpoint for asset '{name}' is not available in the local cache. "
        f"Configure or download the asset before running this stage."
    )
