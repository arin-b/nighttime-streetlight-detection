from rbccps_od.data_management.corpus_v3 import build_negative_rows, build_positive_rows
from rbccps_od.data_management.seed_sources import ImageRecord


def make_record(uid: str, clip_id: str, frame_id: str) -> ImageRecord:
    return ImageRecord(
        image_uid=uid,
        dataset_id=uid.split(":")[0],
        source_export_split="train",
        source_image_path=f"/tmp/{uid}.jpg",  # type: ignore[arg-type]
        source_file_name=f"{uid}.jpg",
        canonical_name=uid,
        clip_id=clip_id,
        frame_id=frame_id,
        width=100,
        height=50,
        has_annotation=True,
        annotation_count=1,
        original_image_id=1,
        output_file_name=f"{uid}.jpg",
    )


def test_build_positive_rows_handles_keep_and_blocks_missing_scene():
    record = make_record("jobin:img1", "clip1", "1")
    rows, manifest, validation, blockers = build_positive_rows(
        records=[record],
        original_boxes={"jobin:img1": [[0.0, 0.0, 10.0, 10.0]]},
        review_rows={"jobin:img1": {"review_status": "keep", "scene_bucket": "quiet_residential_lane", "updated_boxes_json": ""}},
        split_map={"jobin:img1": "valid"},
        allow_unreviewed=False,
        allow_missing_scene=False,
    )
    assert not blockers
    assert rows[0]["assigned_split"] == "valid"
    assert manifest[0]["scene_bucket"] == "quiet_residential_lane"
    assert validation[0]["validation_role"] == "positive"


def test_build_negative_rows_preserves_clean_negative_semantics():
    rows, validation, blockers = build_negative_rows(
        negative_reviews=[{
            "review_candidate_id": "neg1",
            "dataset_id": "jobin",
            "clip_id": "clip1",
            "frame_id": "10",
            "review_status": "clean_negative",
            "scene_bucket": "busy_arterial_road",
            "source_pool": "local",
            "image_path": "/tmp/neg1.jpg",
        }],
        existing_admissions={},
        allow_missing_scene=False,
        valid_target=1,
        test_target=1,
    )
    assert not blockers
    assert rows[0]["corpus_role"] == "reviewed_clean_negative"
    assert rows[0]["assigned_split"] == "valid"
    assert validation[0]["validation_role"] == "background"
