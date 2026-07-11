from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rbccps_measurement.contracts.input_schema import DetectorTrackRecord


def load_detector_tracks_jsonl(path: str | Path) -> tuple[DetectorTrackRecord, ...]:
    records: list[DetectorTrackRecord] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid detector JSONL at line {line_no}: {exc}") from exc
        records.append(DetectorTrackRecord.from_dict(payload))
    return tuple(records)
