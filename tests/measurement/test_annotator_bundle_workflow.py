from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from rbccps_annotator.bundle_workflow import (
    auto_sample_budget,
    prepare_bundle_workspace,
    scan_raw_media,
    validate_tutorial_examples,
)
from rbccps_annotator.exports import export_yolo


def _write_image(path: Path, color: tuple[int, int, int], bright: bool = False) -> None:
    image = Image.new("RGB", (96, 64), color)
    if bright:
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 8, 52, 20), fill=(255, 245, 180))
    image.save(path)


def test_scan_raw_media_and_auto_budget(tmp_path: Path) -> None:
    raw = tmp_path / "input_raw"
    raw.mkdir()
    _write_image(raw / "a.jpg", (10, 10, 10))
    (raw / "clip.mp4").write_bytes(b"not a real video")
    (raw / "notes.txt").write_text("ignore", encoding="utf-8")

    found = scan_raw_media(raw)

    assert [path.name for path in found] == ["a.jpg", "clip.mp4"]
    assert auto_sample_budget(10) == 10
    assert 300 <= auto_sample_budget(1000) <= 600


def test_prepare_bundle_workspace_from_images_and_split_export(tmp_path: Path) -> None:
    raw = tmp_path / "input_raw"
    raw.mkdir()
    for index in range(8):
        _write_image(raw / f"frame_{index}.jpg", (8 + index * 20, 10, 16), bright=index % 2 == 0)
    tutorial = tmp_path / "tutorial_examples"
    tutorial.mkdir()
    _write_image(tutorial / "example.jpg", (20, 20, 20), bright=True)
    (tutorial / "example.json").write_text(
        json.dumps(
            {
                "id": "example",
                "title": "Navigation basics",
                "lesson": "Practice selecting and saving.",
                "image": "example.jpg",
                "review": {
                    "schema_version": "measurement_annotator_v1",
                    "boxes": [{"box_id": "box_001", "class_name": "streetlight_lamp_head", "bbox_xyxy": [40, 8, 52, 20], "status": "accepted"}],
                    "confounder_boxes": [],
                    "polygons": [],
                    "measurement": {
                        "lamp_status": [],
                        "public_space_regions": [],
                        "affected_regions": [],
                        "visibility_labels": [],
                        "attribution_labels": [],
                        "lux_points": [],
                        "qa_flags": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = prepare_bundle_workspace(tmp_path, input_raw=raw, tutorial_examples=tutorial, sample_budget=5, force=True)

    assert result.sampled_count == 5
    assert result.tutorial_count == 1
    manifest = json.loads((result.workspace / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["items"]) == 5
    assert all(item["metadata"]["metadata_quality"] == "video_image_only" for item in manifest["items"])

    export_yolo(result.workspace, result.workspace / "exports" / "yolo_split", split_dirs=True, include_candidate_boxes=True)
    assert (result.workspace / "exports" / "yolo_split" / "images" / "train").exists()
    assert (result.workspace / "exports" / "yolo_split" / "dataset.yaml").exists()


def test_validate_tutorial_rejects_bad_examples(tmp_path: Path) -> None:
    tutorial = tmp_path / "tutorial_examples"
    tutorial.mkdir()
    (tutorial / "bad.json").write_text(json.dumps({"image": "missing.jpg", "review": {"boxes": []}}), encoding="utf-8")

    try:
        validate_tutorial_examples(tutorial, tmp_path / "workspace")
    except ValueError as error:
        assert "invalid" in str(error)
    else:
        raise AssertionError("expected invalid tutorial examples to raise")
