from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_measurement.ingest.pseudo_manifest import PseudoManifestOptions, build_pseudo_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a plausible measurement clip manifest from still images.")
    parser.add_argument("--images", nargs="+", required=True, help="One or more image paths in frame order.")
    parser.add_argument("--out", required=True, help="Output clip_manifest.json path.")
    parser.add_argument("--clip-id", default="pseudo_clip_001", help="Clip ID to write into the manifest.")
    parser.add_argument("--device-id", default="pseudo_android_phone", help="Device ID to write into the manifest.")
    parser.add_argument("--fps", type=float, default=30.0, help="Synthetic frame rate for timestamps.")
    parser.add_argument("--max-lamps-per-frame", type=int, default=2, help="Maximum pseudo lamp boxes per frame.")
    parser.add_argument("--lat", type=float, default=None, help="Optional approximate latitude.")
    parser.add_argument("--lon", type=float, default=None, help="Optional approximate longitude.")
    parser.add_argument("--heading-deg", type=float, default=None, help="Optional approximate camera heading.")
    parser.add_argument("--copy-images", action="store_true", help="Copy input images into an adjacent frames/ directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = PseudoManifestOptions(
        clip_id=args.clip_id,
        device_id=args.device_id,
        fps=args.fps,
        max_lamps_per_frame=args.max_lamps_per_frame,
        latitude=args.lat,
        longitude=args.lon,
        heading_deg=args.heading_deg,
        copy_images=args.copy_images,
    )
    manifest = build_pseudo_manifest([Path(path) for path in args.images], Path(args.out), options)
    print(json.dumps({
        "manifest": str(Path(args.out)),
        "frames": len(manifest.frames),
        "tracks": len(manifest.tracks),
        "source_model": "pseudo_bright_region_estimator_v1",
    }, indent=2))


if __name__ == "__main__":
    main()
