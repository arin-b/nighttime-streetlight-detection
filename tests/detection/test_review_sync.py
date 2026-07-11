from rbccps_od.data_management.review_sync import positive_review_rows, secondary_exports
from rbccps_od.data_management.seed_sources import ImageRecord


def make_record(uid: str, clip_id: str, frame_id: str, has_annotation: bool = True) -> ImageRecord:
    return ImageRecord(
        image_uid=uid,
        dataset_id=uid.split(":")[0],
        source_export_split="train",
        source_image_path=f"/tmp/{uid}.jpg",  # type: ignore[arg-type]
        source_file_name=f"{uid}.jpg",
        canonical_name=uid,
        clip_id=clip_id,
        frame_id=frame_id,
        width=640,
        height=480,
        has_annotation=has_annotation,
        annotation_count=1 if has_annotation else 0,
        original_image_id=1,
    )


def test_positive_review_rows_maps_click_review_state():
    records = [make_record("jobin:img1", "clip1", "1")]
    existing_rows = [{
        "key": "jobin:img1",
        "primary_decision": "fix",
        "secondary_reason": "fix_off_center",
        "scene_bucket": "quiet_residential_lane",
        "updated_boxes_json": "[[1,2,3,4]]",
        "review_timestamp": "2026-01-01 00:00:00",
    }]
    rows = positive_review_rows("jobin_positive", records, existing_rows, {"jobin:img1": "train"})
    assert rows[0]["review_status"] == "fix_box"
    assert rows[0]["fix_reason"] == "fix_off_center"
    assert rows[0]["scene_bucket"] == "quiet_residential_lane"


def test_secondary_exports_collects_scene_and_promotions():
    scene_rows, promoted_rows = secondary_exports(
        [{"mode": "jobin_positive", "key": "a", "dataset_id": "jobin", "clip_id": "c", "frame_id": "1", "scene_bucket": "quiet_residential_lane", "review_status": "keep", "review_timestamp": ""}],
        [],
        [{"mode": "negative_review", "key": "b", "review_candidate_id": "b", "source_pool": "pool", "dataset_id": "jobin", "clip_id": "c", "frame_id": "2", "image_path": "img.jpg", "scene_bucket": "busy_arterial_road", "review_status": "promote_to_positive_review", "review_timestamp": ""}],
    )
    assert len(scene_rows) == 2
    assert promoted_rows[0]["review_candidate_id"] == "b"
