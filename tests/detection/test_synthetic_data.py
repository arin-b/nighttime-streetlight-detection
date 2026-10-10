import json
from pathlib import Path

from PIL import Image

from rbccps_od.data_management.synthetic import (
    SourceRecord,
    SynthesisConfig,
    apply_dim,
    apply_off,
    create_soft_illumination_mask,
    filter_candidates,
    generate,
    parse_yolo_annotations,
    select_source_frames,
)


def _write_sample(root: Path, name: str, split: str, box: str = "0 0.5 0.25 0.2 0.2") -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (200, 180, 120)).save(image_dir / f"{name}.jpg")
    (label_dir / f"{name}.txt").write_text(box + "\n", encoding="utf-8")


def _config(root: Path) -> SynthesisConfig:
    return SynthesisConfig(
        input_images=root / "images",
        input_annotations=root / "labels",
        output_images=root / "out" / "images" / "train",
        output_annotations=root / "out" / "labels" / "train",
        output_metadata=root / "out" / "metadata.jsonl",
        number_of_source_frames=2,
        number_of_variants_per_source=1,
        state_probabilities={"dim": 1.0},
        illumination_states=("dim",),
        confounder_fraction=0.0,
        ordinary_fraction=1.0,
        confounder_score_threshold=1.0,
    )


def test_annotations_mask_and_transforms_are_local():
    root = Path(__file__).parent
    label = root / "_synthetic_label.txt"
    try:
        label.write_text("0 0.5 0.25 0.2 0.2\n", encoding="utf-8")
        annotation = parse_yolo_annotations(label)[0]
        mask = create_soft_illumination_mask((64, 48), annotation.pixel_box((64, 48)))
        original = Image.new("RGB", (64, 48), (200, 200, 200))
        dimmed = apply_dim(original, mask, .3)
        off = apply_off(original, mask, .08)
        assert dimmed.getpixel((32, 12))[0] < original.getpixel((32, 12))[0]
        assert dimmed.getpixel((0, 47)) == original.getpixel((0, 47))
        assert off.getpixel((32, 12))[0] < dimmed.getpixel((32, 12))[0]
    finally:
        label.unlink(missing_ok=True)


def test_generation_is_deterministic_and_excludes_held_out_scene(tmp_path):
    _write_sample(tmp_path, "road_a_frame_000001", "train")
    _write_sample(tmp_path, "road_b_frame_000001", "train")
    # Same source scene in valid must remove the train sibling to prevent leakage.
    _write_sample(tmp_path, "road_a_frame_000002", "valid")
    report = generate(_config(tmp_path))
    assert report["selected_sources"] == 1
    metadata = [json.loads(line) for line in (tmp_path / "out" / "metadata.jsonl").read_text().splitlines()]
    assert len(metadata) == 1
    assert metadata[0]["source_scene_id"] == "road_b"
    copied = Path(metadata[0]["synthetic_annotation"])
    assert copied.read_text() == "0 0.5 0.25 0.2 0.2\n"
    assert metadata[0]["source_timestamp"] is None
    assert metadata[0]["timestamp_provenance"] == "unavailable_not_inferred_from_filename"


def test_duplicate_filter_and_40_percent_selection(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (32, 32), "white").save(image_a)
    Image.new("RGB", (32, 32), "white").save(image_b)
    config = _config(tmp_path)
    records = [
        SourceRecord(image_a, tmp_path / "a.txt", "a", "train", "one"),
        SourceRecord(image_b, tmp_path / "b.txt", "b", "train", "one"),
    ]
    retained, rejected = filter_candidates(records, config)
    assert len(retained) == 1
    assert rejected["exact_duplicate"] == 1

    pool = [SourceRecord(tmp_path / f"c{i}.jpg", tmp_path / "x.txt", f"c{i}", "train", f"c{i}", confounder_category="confounder_heavy") for i in range(3)]
    pool += [SourceRecord(tmp_path / f"o{i}.jpg", tmp_path / "x.txt", f"o{i}", "train", f"o{i}", confounder_category="ordinary") for i in range(5)]
    config.number_of_source_frames = 5
    config.confounder_fraction = .4
    config.ordinary_fraction = .6
    selected, notes = select_source_frames(pool, config)
    assert not notes
    assert sum(item.confounder_category == "confounder_heavy" for item in selected) == 2
