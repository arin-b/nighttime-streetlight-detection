from __future__ import annotations

import argparse
import json
from pathlib import Path

from rbccps_measurement.dataset_prep.converter import INPUT_TYPES, VALIDATION_MODES, convert_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert draft/reviewed measurement annotations into a prepared dataset layout.")
    parser.add_argument("--input", required=True, type=Path, help="Annotation JSON, annotator workspace, measurement export directory, or directory of inputs.")
    parser.add_argument("--input-type", required=True, choices=sorted(INPUT_TYPES), help="Input source type.")
    parser.add_argument("--out", required=True, type=Path, help="Prepared dataset root to write.")
    parser.add_argument("--validation-mode", default="warn", choices=sorted(VALIDATION_MODES), help="Validation behavior for questionable rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = convert_annotations(args.input, args.input_type, args.out, args.validation_mode)
    print(
        json.dumps(
            {
                "output_root": str(args.out),
                "input_type": args.input_type,
                "validation_mode": args.validation_mode,
                "frames": len(data.frames),
                "tracks": len(data.tracks),
                "warnings": len(data.issues),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
