from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from rbccps_od.config.paths import ensure_dir, model_cache_root, repo_root
from rbccps_od.models.registry import get_asset, get_registry


def cached_asset_path(name: str) -> Path:
    spec = get_asset(name)
    return model_cache_root() / spec.name / spec.version / spec.filename


def verify_checksum(path: Path, checksum: str) -> bool:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == checksum


def _extract_archive_member(archive_path: Path, member_name: str, target: Path) -> Path:
    ensure_dir(target.parent)
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open(member_name) as src, target.open("wb") as dst:
            dst.write(src.read())
    return target


def download_asset(name: str) -> Path:
    spec = get_asset(name)
    target = cached_asset_path(name)
    if spec.local_path:
        local_target = Path(spec.local_path)
        if not local_target.is_absolute():
            local_target = (repo_root() / local_target).resolve()
        if local_target.exists():
            return local_target
    if target.exists():
        if spec.checksum and not verify_checksum(target, spec.checksum):
            raise RuntimeError(f"Checksum mismatch for cached asset {name} at {target}")
        return target
    archive_asset = spec.metadata.get("archive_asset")
    archive_member = spec.metadata.get("archive_member")
    if archive_asset and archive_member:
        archive_path = download_asset(str(archive_asset))
        return _extract_archive_member(archive_path, str(archive_member), target)
    if not spec.url:
        raise RuntimeError(f"Model asset '{name}' has no configured download URL. Fill config/model_registry.py before downloading.")
    ensure_dir(target.parent)
    urllib.request.urlretrieve(spec.url, target)
    if spec.checksum and not verify_checksum(target, spec.checksum):
        raise RuntimeError(f"Checksum mismatch after download for {name}")
    return target


def download_all() -> dict[str, str]:
    results: dict[str, str] = {}
    for name in get_registry():
        try:
            results[name] = str(download_asset(name))
        except Exception as exc:
            results[name] = f"unavailable: {exc}"
    return results


def main() -> None:
    print(json.dumps(download_all(), indent=2))


if __name__ == "__main__":
    main()
