from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.ingest.validation import validate_clip_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a measurement-block dataset manifest directory.")
    parser.add_argument("--manifest", required=True, help="JSON file containing a list of clip manifest paths or clip manifest objects.")
    parser.add_argument("--out", required=True, help="Output dataset root.")
    parser.add_argument("--copy-manifests", action="store_true", help="Copy clip manifests into clips/ instead of referencing them.")
    return parser.parse_args()


def _load_clip_entries(path: Path) -> list[dict[str, object] | str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "clips" in payload:
        return list(payload["clips"])
    if isinstance(payload, list):
        return payload
    raise ValueError("dataset manifest must be a list or an object with a 'clips' list")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out = Path(args.out)
    clips_dir = out / "clips"
    for subdir in [
        clips_dir,
        out / "tracks",
        out / "frames",
        out / "annotations" / "semantic_masks",
        out / "annotations" / "affected_regions",
        out / "annotations" / "confounders",
        out / "lux",
        out / "splits",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)

    entries = _load_clip_entries(manifest_path)
    prepared = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            source_path = (manifest_path.parent / entry).resolve() if not Path(entry).is_absolute() else Path(entry)
            clip = ClipManifest.load(source_path)
            validate_clip_manifest(clip)
            if args.copy_manifests:
                target = clips_dir / f"{clip.clip_id}.json"
                shutil.copy2(source_path, target)
                ref = str(target.relative_to(out)).replace("\\", "/")
            else:
                ref = str(source_path)
        elif isinstance(entry, dict):
            clip = ClipManifest.from_dict(entry)
            validate_clip_manifest(clip)
            target = clips_dir / f"{clip.clip_id}.json"
            target.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            ref = str(target.relative_to(out)).replace("\\", "/")
        else:
            raise ValueError(f"unsupported clip entry at index {index}: {type(entry)!r}")
        prepared.append({
            "clip_id": clip.clip_id,
            "device_id": clip.device_id,
            "calibration_level": clip.calibration_level,
            "manifest": ref,
            "num_frames": len(clip.frames),
            "num_tracks": len(clip.tracks),
        })

    dataset_manifest = {
        "dataset_type": "rbccps_measurement",
        "version": "pilot_v1_skeleton",
        "source_manifest": str(manifest_path),
        "clips": prepared,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(out), "clips": len(prepared)}, indent=2))


if __name__ == "__main__":
    main()
