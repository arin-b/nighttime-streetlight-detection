from __future__ import annotations

from collections import defaultdict

from rbccps_measurement.contracts.input_schema import DetectorTrackRecord


def group_track_crops(tracks: tuple[DetectorTrackRecord, ...]) -> dict[str, list[DetectorTrackRecord]]:
    grouped: dict[str, list[DetectorTrackRecord]] = defaultdict(list)
    for track in tracks:
        grouped[track.track_id].append(track)
    return {key: sorted(value, key=lambda item: item.timestamp_ns) for key, value in grouped.items()}
