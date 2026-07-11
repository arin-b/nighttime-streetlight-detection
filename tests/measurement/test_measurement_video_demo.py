from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.ingest.validation import validate_clip_manifest
from rbccps_measurement.pipeline import run_clip_to_directory
from scripts.measurement.run_measurement_video_demo import (
    FrameDetection,
    LinkedTrack,
    bbox_iou,
    build_clip_manifest_payload,
    link_detections,
)


def test_bbox_iou_returns_expected_overlap() -> None:
    value = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert round(value, 4) == 0.1429
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_link_detections_keeps_same_track_for_overlapping_boxes() -> None:
    tracks = link_detections(
        [
            [FrameDetection(frame_index=1, bbox_xyxy=(10, 10, 30, 30), score=0.9)],
            [FrameDetection(frame_index=2, bbox_xyxy=(11, 10, 31, 30), score=0.8)],
        ],
        iou_threshold=0.3,
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == "demo_lamp_0001"
    assert [item.frame_index for item in tracks[0].detections] == [1, 2]


def test_link_detections_creates_new_track_for_far_boxes() -> None:
    tracks = link_detections(
        [
            [FrameDetection(frame_index=1, bbox_xyxy=(10, 10, 30, 30), score=0.9)],
            [FrameDetection(frame_index=2, bbox_xyxy=(80, 80, 100, 100), score=0.8)],
        ],
        iou_threshold=0.3,
    )

    assert [track.track_id for track in tracks] == ["demo_lamp_0001", "demo_lamp_0002"]
    assert [len(track.detections) for track in tracks] == [1, 1]


def test_demo_manifest_builder_validates_and_runs_measurement(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_frames"
    source_dir.mkdir()
    frame_paths: list[Path] = []
    for index in range(1, 3):
        frame_path = source_dir / f"frame_{index:06d}.jpg"
        Image.new("RGB", (96, 72), (12 + index, 12, 10)).save(frame_path)
        frame_paths.append(frame_path)

    linked_tracks = [
        LinkedTrack(
            track_id="demo_lamp_0001",
            last_bbox=(35, 10, 50, 25),
            last_frame_index=2,
            detections=[
                FrameDetection(frame_index=1, bbox_xyxy=(34, 10, 49, 25), score=0.86),
                FrameDetection(frame_index=2, bbox_xyxy=(35, 10, 50, 25), score=0.87),
            ],
        )
    ]
    out = tmp_path / "demo"
    manifest = build_clip_manifest_payload(frame_paths, linked_tracks, out, fps=3.0)

    validate_clip_manifest(ClipManifest.from_dict(manifest))
    manifest_path = out / "clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    reports = run_clip_to_directory(manifest_path, out / "measurement", measurement_run_id="test_demo")

    assert len(reports) == 1
    assert reports[0].lamp_track_id == "demo_lamp_0001"
    assert (out / "measurement" / "reports.json").exists()
