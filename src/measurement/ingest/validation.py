from __future__ import annotations

from rbccps_measurement.contracts.input_schema import ACCEPTED_BBOX_FORMATS, ACCEPTED_TRACK_CLASSES, ClipManifest


def validate_clip_manifest(manifest: ClipManifest) -> None:
    if not manifest.frames:
        raise ValueError("clip manifest must contain at least one frame")
    if not manifest.tracks:
        raise ValueError("clip manifest must contain detector tracks; missing detections are not interpreted as off lamps")

    frames = manifest.frame_by_id()
    timestamps = {frame.timestamp_ns for frame in manifest.frames}
    if len(timestamps) != len(manifest.frames):
        raise ValueError("frame timestamps must be unique")
    frame_ids = {frame.frame_id for frame in manifest.frames}
    if len(frame_ids) != len(manifest.frames):
        raise ValueError("frame IDs must be unique")
    sorted_frame_times = sorted(frame.timestamp_ns for frame in manifest.frames)
    if any(later <= earlier for earlier, later in zip(sorted_frame_times, sorted_frame_times[1:])):
        raise ValueError("frame timestamps must be strictly increasing after sorting")

    for track in manifest.tracks:
        if track.class_name not in ACCEPTED_TRACK_CLASSES:
            raise ValueError(f"track {track.track_id} has unsupported class {track.class_name!r}")
        if track.bbox_format not in ACCEPTED_BBOX_FORMATS:
            raise ValueError(f"track {track.track_id} has unsupported bbox_format {track.bbox_format!r}")
        frame = frames.get(track.frame_id)
        if frame is None:
            raise ValueError(f"track {track.track_id} references missing frame_id {track.frame_id}")
        if frame.timestamp_ns != track.timestamp_ns:
            raise ValueError(
                f"track {track.track_id} timestamp does not match frame {track.frame_id}: "
                f"{track.timestamp_ns} != {frame.timestamp_ns}"
            )
        x1, y1, x2, y2 = track.bbox_xyxy
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"track {track.track_id} bbox must be xyxy with positive width and height")
        if track.bbox_format == "pixel_xyxy_original_frame":
            if x1 < 0 or y1 < 0 or x2 > frame.width or y2 > frame.height:
                raise ValueError(f"track {track.track_id} pixel bbox is outside original frame bounds")
        else:
            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                raise ValueError(f"track {track.track_id} normalized bbox must be within [0, 1]")

    track_seen: dict[str, set[int]] = {}
    track_times: dict[str, list[int]] = {}
    for track in manifest.tracks:
        track_seen.setdefault(track.track_id, set())
        if track.frame_id in track_seen[track.track_id]:
            raise ValueError(f"track {track.track_id} has duplicate detection for frame_id {track.frame_id}")
        track_seen[track.track_id].add(track.frame_id)
        track_times.setdefault(track.track_id, []).append(track.timestamp_ns)
    for track_id, times in track_times.items():
        sorted_times = sorted(times)
        if any(later <= earlier for earlier, later in zip(sorted_times, sorted_times[1:])):
            raise ValueError(f"track {track_id} timestamps must be strictly increasing")
