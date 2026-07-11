from pathlib import Path

from rbccps_od.pipeline.advanced_runner import group_sequence_paths, infer_frame_index


def test_infer_frame_index_prefers_frame_suffix():
    assert infer_frame_index(Path("jobin__val_set_1_frame_22.jpg")) == 22
    assert infer_frame_index(Path("neg__neg_0110__0044.jpg")) == 44


def test_group_sequence_paths_groups_by_prefix_and_sorts_frames():
    paths = [
        Path("jobin__val_set_1_frame_10.jpg"),
        Path("jobin__val_set_1_frame_2.jpg"),
        Path("jobin__test_set_1_frame_1.jpg"),
        Path("neg__neg_0110__0044.jpg"),
    ]
    groups = group_sequence_paths(paths)
    assert list(groups) == ["jobin__test_set_1", "jobin__val_set_1", "neg__neg_0110__0044"]
    assert [p.name for p in groups["jobin__val_set_1"]] == [
        "jobin__val_set_1_frame_2.jpg",
        "jobin__val_set_1_frame_10.jpg",
    ]
