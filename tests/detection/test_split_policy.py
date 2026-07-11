from rbccps_od.data_management.seed_sources import ImageRecord
from rbccps_od.data_management.split_policy import assign_clip_splits


def make_record(dataset_id: str, clip_id: str, idx: int) -> ImageRecord:
    return ImageRecord(
        image_uid=f"{dataset_id}:{clip_id}:{idx}",
        dataset_id=dataset_id,
        source_export_split="train",
        source_image_path=None,  # type: ignore[arg-type]
        source_file_name=f"{clip_id}_frame_{idx}.jpg",
        canonical_name=f"{clip_id}_frame_{idx}",
        clip_id=clip_id,
        frame_id=str(idx),
        width=1280,
        height=720,
        has_annotation=True,
        annotation_count=1,
        original_image_id=idx,
    )


def test_assign_clip_splits_is_clip_safe():
    records = []
    for dataset in ("jobin", "arindam"):
        for clip in ("clip_a", "clip_b", "clip_c", "clip_d"):
            for idx in range(3):
                records.append(make_record(dataset, clip, idx))

    split_map = assign_clip_splits(records)

    seen = {}
    for record in records:
        key = f"{record.dataset_id}:{record.clip_id}"
        split = split_map[key]
        if key in seen:
            assert seen[key] == split
        else:
            seen[key] = split
    assert {v for v in split_map.values()} <= {"train", "valid", "test"}
