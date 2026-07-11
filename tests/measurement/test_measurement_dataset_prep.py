import csv
import json
from pathlib import Path

import pytest

from rbccps_measurement.dataset_prep.converter import convert_annotations
from rbccps_measurement.dataset_prep.normalization import (
    normalize_attribution,
    normalize_confounder,
    normalize_visibility,
)
from rbccps_measurement.training.readiness import check_dataset_readiness


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_normalizers_map_draft_label_drift_and_warn_unknowns():
    assert normalize_visibility("limited").normalized == "marginal"
    assert normalize_visibility("partial").normalized == "marginal"
    unknown_visibility = normalize_visibility("barely-visible-ish")
    assert unknown_visibility.normalized == "unknown"
    assert unknown_visibility.warning

    assert normalize_attribution("streetlight_primary").normalized == "certain"
    assert normalize_attribution("unclear").normalized == "uncertain"
    unknown_attr = normalize_attribution("mostly_magic")
    assert unknown_attr.normalized == "uncertain"
    assert unknown_attr.warning

    assert normalize_confounder("vehicle_headlights").normalized == "headlight"
    assert normalize_confounder("bright_shopfront_window").normalized == "shopfront_or_window"
    assert normalize_confounder("wet_road_reflection").normalized == "reflection"
    assert normalize_confounder("traffic_signal").normalized == "sign_or_signal"


def test_llm_json_conversion_writes_prepared_dataset_and_warnings(tmp_path: Path):
    source = write_json(
        tmp_path / "annotation_llm.json",
        {
            "dataset_zip": "batch_001.zip",
            "annotations": [
                {
                    "image_name": "frame_001.jpg",
                    "width": 100,
                    "height": 80,
                    "boxes": [
                        {
                            "box_id": "box_001",
                            "class_name": "streetlight_lamp_head",
                            "bbox_xyxy": [-5, 10, 30, 40],
                            "track_id": "track_001",
                            "source": "fixture",
                        }
                    ],
                    "confounder_boxes": [
                        {
                            "box_id": "other_001",
                            "surface_type": "vehicle_headlights",
                            "bbox_xyxy": [50, 50, 75, 70],
                        }
                    ],
                    "measurement": {
                        "lamp_status": [{"track_id": "track_001", "status": "on"}],
                        "public_space_regions": [{"region_type": "sidewalk", "points": [[0, 80], [20, 80], [20, 50]]}],
                        "affected_regions": [{"track_id": "track_001", "region_type": "road", "points": [[0, 80], [50, 80], [40, 55]]}],
                        "visibility_labels": [{"track_id": "track_001", "visibility_class": "limited"}],
                        "attribution_labels": [{"track_id": "track_001", "attribution_class": "streetlight_primary"}],
                        "lux_points": [],
                        "qa_flags": [{"flag": "needs_human_verification"}],
                    },
                }
            ],
        },
    )

    out = tmp_path / "prepared"
    convert_annotations(source, "llm-json", out)

    assert (out / "dataset_manifest.json").exists()
    assert (out / "clips" / "batch_001.json").exists()
    assert read_csv(out / "annotations" / "visibility_labels.csv")[0]["visibility_class"] == "marginal"
    assert read_csv(out / "annotations" / "attribution_labels.csv")[0]["attribution_class"] == "certain"
    assert read_csv(out / "annotations" / "confounders.csv")[0]["confounder_type"] == "headlight"
    assert read_csv(out / "tracks" / "tracks.csv")[0]["bbox_xyxy"] == "[0.0, 10.0, 30.0, 40.0]"
    assert read_csv(out / "validation" / "warnings.csv")
    assert read_csv(out / "lux" / "lux_points.csv") == []


def test_annotator_workspace_conversion_uses_reviews(tmp_path: Path):
    workspace = tmp_path / "workspace"
    write_json(
        workspace / "manifest.json",
        {
            "schema_version": "measurement_annotator_v1",
            "workspace_id": "ws",
            "items": [
                {
                    "key": "item_001",
                    "image_path": str(tmp_path / "frame.jpg"),
                    "width": 64,
                    "height": 48,
                    "clip_id": "clip_ws",
                    "frame_id": "7",
                }
            ],
        },
    )
    write_json(
        workspace / "reviews" / "items" / "item_001.json",
        {
            "boxes": [{"track_id": "track_a", "bbox_xyxy": [1, 2, 10, 12], "class_name": "streetlight"}],
            "confounder_boxes": [{"surface_type": "shopfront_or_sign_lighting", "bbox_xyxy": [20, 20, 30, 30]}],
            "measurement": {
                "visibility_labels": [{"track_id": "track_a", "visibility_class": "partial"}],
                "attribution_labels": [{"track_id": "track_a", "attribution_class": "unclear"}],
            },
        },
    )

    out = tmp_path / "prepared_ws"
    convert_annotations(workspace, "annotator-workspace", out)

    assert read_csv(out / "tracks" / "tracks.csv")[0]["class_name"] == "streetlight_lamp_head"
    assert read_csv(out / "annotations" / "visibility_labels.csv")[0]["visibility_class"] == "marginal"
    assert read_csv(out / "annotations" / "attribution_labels.csv")[0]["attribution_class"] == "uncertain"
    assert read_csv(out / "annotations" / "confounders.csv")[0]["confounder_type"] == "shopfront_or_window"


def test_measurement_export_conversion_preserves_provenance(tmp_path: Path):
    export = tmp_path / "measurement_export"
    export.mkdir()
    (export / "tracks.csv").write_text(
        "key,clip_id,frame_id,timestamp_ns,image_path,width,height,track_id,class_name,bbox_xyxy\n"
        "k1,clip_csv,1,1000,frame.jpg,80,60,t1,streetlight_lamp_head,\"[2, 3, 12, 18]\"\n",
        encoding="utf-8",
    )
    (export / "visibility_labels.csv").write_text(
        "key,clip_id,frame_id,track_id,visibility_class\n"
        "k1,clip_csv,1,t1,limited\n",
        encoding="utf-8",
    )
    (export / "attribution_labels.csv").write_text(
        "key,clip_id,frame_id,track_id,attribution_class\n"
        "k1,clip_csv,1,t1,streetlight_primary\n",
        encoding="utf-8",
    )

    out = tmp_path / "prepared_csv"
    convert_annotations(export, "measurement-export", out)

    row = read_csv(out / "annotations" / "visibility_labels.csv")[0]
    assert row["source_file"].endswith("visibility_labels.csv")
    assert row["source_item"] == "k1"
    assert row["raw_label"] == "limited"
    assert row["normalized_label"] == "marginal"


def test_validation_modes_quarantine_or_fail(tmp_path: Path):
    source = write_json(
        tmp_path / "bad_annotation_llm.json",
        {
            "dataset_zip": "bad.zip",
            "annotations": [
                {
                    "image_name": "frame.jpg",
                    "width": 40,
                    "height": 40,
                    "boxes": [{"track_id": "track_1", "bbox_xyxy": [1, 1, 10, 10]}],
                    "measurement": {
                        "visibility_labels": [{"track_id": "missing_track", "visibility_class": "adequate"}],
                    },
                }
            ],
        },
    )

    quarantine_out = tmp_path / "quarantine"
    convert_annotations(source, "llm-json", quarantine_out, validation_mode="quarantine")
    assert read_csv(quarantine_out / "annotations" / "visibility_labels.csv") == []
    assert read_csv(quarantine_out / "validation" / "invalid_rows.csv")

    with pytest.raises(ValueError, match="missing track"):
        convert_annotations(source, "llm-json", tmp_path / "fail", validation_mode="fail")


def test_prepared_dataset_is_readiness_compatible(tmp_path: Path):
    source = write_json(
        tmp_path / "annotation_llm.json",
        {
            "dataset_zip": "ready.zip",
            "annotations": [
                {
                    "image_name": "frame.jpg",
                    "width": 50,
                    "height": 50,
                    "boxes": [{"track_id": "track_1", "bbox_xyxy": [5, 5, 15, 15]}],
                    "measurement": {},
                }
            ],
        },
    )
    out = tmp_path / "ready_dataset"
    convert_annotations(source, "llm-json", out)

    report = check_dataset_readiness(out, require_annotations=False, ensure_models=False)
    assert report.ready is True
    assert report.clips == 1
    assert report.frames == 1
    assert report.tracks == 1
