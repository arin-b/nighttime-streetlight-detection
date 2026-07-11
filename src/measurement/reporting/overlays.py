from __future__ import annotations

import json
from pathlib import Path

from rbccps_measurement.contracts.output_schema import MeasurementReport


def write_overlay_manifest(path: str | Path, reports: list[MeasurementReport]) -> None:
    """Write a lightweight overlay manifest.

    Real image/video rendering plugs into this file later. The manifest keeps
    the demo reproducible without requiring OpenCV/Pillow in base tests.
    """

    payload = {
        "overlay_type": "measurement_debug_manifest",
        "items": [
            {
                "lamp_track_id": report.lamp_track_id,
                "mask_uri": report.affected_region["image_mask_uri"],
                "category": report.metrics["overall_category"],
                "flags": report.uncertainty_flags,
            }
            for report in reports
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
