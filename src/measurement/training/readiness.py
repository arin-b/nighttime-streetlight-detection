from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.ingest.validation import validate_clip_manifest
from rbccps_measurement.models.downloader import cached_asset_path, ensure_required_assets
from rbccps_measurement.models.registry import get_registry


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    dataset_root: str
    clips: int
    frames: int
    tracks: int
    issues: list[str]
    model_assets: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "dataset_root": self.dataset_root,
            "clips": self.clips,
            "frames": self.frames,
            "tracks": self.tracks,
            "issues": self.issues,
            "model_assets": self.model_assets,
        }


def _load_prepared_dataset(dataset_root: Path) -> list[Path]:
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Prepared dataset is missing dataset_manifest.json: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    clip_paths: list[Path] = []
    for item in payload.get("clips", []):
        manifest_ref = Path(item["manifest"])
        clip_paths.append(manifest_ref if manifest_ref.is_absolute() else dataset_root / manifest_ref)
    return clip_paths


def check_dataset_readiness(dataset_root: str | Path, require_annotations: bool = True, ensure_models: bool = True) -> ReadinessReport:
    root = Path(dataset_root)
    issues: list[str] = []
    frames = 0
    tracks = 0
    clips = 0
    try:
        clip_paths = _load_prepared_dataset(root)
    except Exception as exc:
        return ReadinessReport(False, str(root), 0, 0, 0, [str(exc)], {})

    if not clip_paths:
        issues.append("dataset_manifest.json contains no clips")

    for clip_path in clip_paths:
        if not clip_path.exists():
            issues.append(f"clip manifest missing: {clip_path}")
            continue
        try:
            clip = ClipManifest.load(clip_path)
            validate_clip_manifest(clip)
            clips += 1
            frames += len(clip.frames)
            tracks += len(clip.tracks)
        except Exception as exc:
            issues.append(f"{clip_path}: {exc}")

    if require_annotations:
        expected_dirs = [
            root / "annotations" / "semantic_masks",
            root / "annotations" / "affected_regions",
            root / "annotations" / "confounders",
            root / "lux",
            root / "splits",
        ]
        for expected in expected_dirs:
            if not expected.exists():
                issues.append(f"required dataset directory missing: {expected}")
        split_files = list((root / "splits").glob("*.json")) if (root / "splits").exists() else []
        if not split_files:
            issues.append("no route/night/device split JSON files found under splits/")

    model_assets: dict[str, str] = {}
    if ensure_models:
        model_assets = ensure_required_assets()
        for name, result in model_assets.items():
            if result.startswith("unavailable:"):
                issues.append(f"required model asset unavailable: {name}: {result}")
    else:
        for name, spec in get_registry().items():
            if spec.required_for_training:
                model_assets[name] = str(cached_asset_path(name))

    return ReadinessReport(
        ready=not issues,
        dataset_root=str(root),
        clips=clips,
        frames=frames,
        tracks=tracks,
        issues=issues,
        model_assets=model_assets,
    )
