from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


from rbccps_od.config.paths import repo_root


ROOT = repo_root()
DATASETS_ROOT = ROOT / "datasets"
OUTPUT_ROOT = DATASETS_ROOT / "derived" / "annotation_automation"
SEED = 20260514

CLIP_RE = re.compile(r"^(?P<clip>(?:.*?_set_\d+|set_\d+))_frame_(?P<frame>\d+)(?:_jpg)?$")


@dataclass
class ImageRecord:
    image_uid: str
    dataset_id: str
    source_export_split: str
    source_image_path: Path
    source_file_name: str
    canonical_name: str
    clip_id: str
    frame_id: str
    width: int
    height: int
    has_annotation: bool
    annotation_count: int
    original_image_id: int
    assigned_split: str = ""
    corpus_role: str = ""
    output_file_name: str = ""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def strip_rf_suffix(file_name: str) -> str:
    return re.sub(r"\.rf\.[^.]+(?=\.[^.]+$)", "", file_name)


def canonical_stem(file_name: str) -> str:
    return Path(strip_rf_suffix(file_name)).stem


def parse_clip_and_frame(file_name: str) -> tuple[str, str]:
    stem = canonical_stem(file_name)
    match = CLIP_RE.match(stem)
    if match:
        return match.group("clip"), match.group("frame")
    fallback = re.match(r"^(?P<clip>.+?)_frame_(?P<frame>\d+)", stem)
    if fallback:
        return fallback.group("clip"), fallback.group("frame")
    return "unknown_clip", stem


def output_file_name(dataset_id: str, source_file_name: str) -> str:
    stem = canonical_stem(source_file_name)
    if stem.endswith("_jpg"):
        stem = stem[:-4]
    return f"{dataset_id}__{stem}.jpg"


def normalized_annotation_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_c = (x + (w / 2.0)) / width
    y_c = (y + (h / 2.0)) / height
    return x_c, y_c, w / width, h / height


def coerce_bbox(bbox: list[float]) -> list[float]:
    return [float(value) for value in bbox]


def sample_evenly(items: list[dict], target_count: int) -> list[dict]:
    if target_count <= 0 or not items:
        return []
    if len(items) <= target_count:
        return items
    indexes = []
    for i in range(target_count):
        raw_index = math.floor(i * len(items) / target_count)
        indexes.append(raw_index)
    picked = []
    seen = set()
    for idx in indexes:
        idx = min(idx, len(items) - 1)
        key = items[idx]["image_path"]
        if key in seen:
            continue
        seen.add(key)
        picked.append(items[idx])
    if len(picked) < target_count:
        for item in items:
            if item["image_path"] in seen:
                continue
            picked.append(item)
            seen.add(item["image_path"])
            if len(picked) >= target_count:
                break
    return picked


def choose_split_counts(clip_count: int) -> tuple[int, int]:
    valid_count = max(1, round(clip_count * 0.15))
    test_count = max(1, round(clip_count * 0.15))
    while valid_count + test_count > clip_count - 1:
        if valid_count >= test_count and valid_count > 1:
            valid_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    return valid_count, test_count


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_jobin_source() -> tuple[list[ImageRecord], list[dict]]:
    source_root = DATASETS_ROOT / "annotated_seed" / "jobin-original-annotated-images"
    records: list[ImageRecord] = []
    annotations: list[dict] = []
    for split_name in ("train", "valid", "test"):
        coco_path = source_root / split_name / "_annotations.coco.json"
        coco = load_json(coco_path)
        ann_counter = Counter(ann["image_id"] for ann in coco["annotations"])
        image_lookup = {image["id"]: image for image in coco["images"]}
        for image in coco["images"]:
            clip_id, frame_id = parse_clip_and_frame(image["file_name"])
            image_uid = f"jobin:{canonical_stem(image['file_name'])}"
            record = ImageRecord(
                image_uid=image_uid,
                dataset_id="jobin",
                source_export_split=split_name,
                source_image_path=source_root / split_name / image["file_name"],
                source_file_name=image["file_name"],
                canonical_name=canonical_stem(image["file_name"]),
                clip_id=clip_id,
                frame_id=frame_id,
                width=image["width"],
                height=image["height"],
                has_annotation=ann_counter[image["id"]] > 0,
                annotation_count=ann_counter[image["id"]],
                original_image_id=image["id"],
                corpus_role="seed_positive" if ann_counter[image["id"]] > 0 else "review_candidate",
                output_file_name=output_file_name("jobin", image["file_name"]),
            )
            records.append(record)
        for ann in coco["annotations"]:
            image = image_lookup[ann["image_id"]]
            annotations.append(
                {
                    "image_uid": f"jobin:{canonical_stem(image['file_name'])}",
                    "dataset_id": "jobin",
                    "bbox": ann["bbox"],
                    "source_category_id": ann["category_id"],
                }
            )
    return records, annotations


def load_arindam_source() -> tuple[list[ImageRecord], list[dict]]:
    source_root = DATASETS_ROOT / "annotated_seed" / "arindam-annotated-images" / "nighttime-streetlight-detection.coco" / "train"
    coco_path = source_root / "_annotations.coco.json"
    coco = load_json(coco_path)
    ann_counter = Counter(ann["image_id"] for ann in coco["annotations"])
    image_lookup = {image["id"]: image for image in coco["images"]}
    records: list[ImageRecord] = []
    annotations: list[dict] = []
    for image in coco["images"]:
        clip_id, frame_id = parse_clip_and_frame(image["file_name"])
        record = ImageRecord(
            image_uid=f"arindam:{canonical_stem(image['file_name'])}",
            dataset_id="arindam",
            source_export_split="train",
            source_image_path=source_root / image["file_name"],
            source_file_name=image["file_name"],
            canonical_name=canonical_stem(image["file_name"]),
            clip_id=clip_id,
            frame_id=frame_id,
            width=image["width"],
            height=image["height"],
            has_annotation=ann_counter[image["id"]] > 0,
            annotation_count=ann_counter[image["id"]],
            original_image_id=image["id"],
            corpus_role="seed_positive" if ann_counter[image["id"]] > 0 else "review_candidate",
            output_file_name=output_file_name("arindam", image["file_name"]),
        )
        records.append(record)
    for ann in coco["annotations"]:
        image = image_lookup[ann["image_id"]]
        annotations.append(
            {
                "image_uid": f"arindam:{canonical_stem(image['file_name'])}",
                "dataset_id": "arindam",
                "bbox": ann["bbox"],
                "source_category_id": ann["category_id"],
            }
        )
    return records, annotations


def assign_clip_splits(records: list[ImageRecord]) -> dict[str, str]:
    split_map: dict[str, str] = {}
    records_by_dataset: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        records_by_dataset[record.dataset_id][record.clip_id].append(record)
    for source_index, dataset_id in enumerate(sorted(records_by_dataset)):
        clips = sorted(records_by_dataset[dataset_id])
        rng = random.Random(SEED + source_index)
        rng.shuffle(clips)
        valid_count, test_count = choose_split_counts(len(clips))
        valid_clips = set(clips[:valid_count])
        test_clips = set(clips[valid_count : valid_count + test_count])
        for clip_id in clips:
            if clip_id in valid_clips:
                split = "valid"
            elif clip_id in test_clips:
                split = "test"
            else:
                split = "train"
            split_map[f"{dataset_id}:{clip_id}"] = split
    return split_map


def copy_image(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)


def build_cleaned_outputs(records: list[ImageRecord], annotations: list[dict]) -> tuple[Path, list[dict], list[dict]]:
    cleaned_root = OUTPUT_ROOT / "yolo_dataset"
    ensure_dir(cleaned_root)
    by_uid: dict[str, ImageRecord] = {record.image_uid: record for record in records}
    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_uid"]].append(ann)

    coco_by_split = {
        split: {
            "info": {"description": "Cleaned merged streetlight seed corpus", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "streetlight", "supercategory": "light_source"}],
        }
        for split in ("train", "valid", "test")
    }
    annotation_metadata_rows: list[dict] = []
    merged_manifest_rows: list[dict] = []
    next_image_id = 1
    next_ann_id = 1

    for record in sorted(records, key=lambda item: (item.assigned_split, item.dataset_id, item.clip_id, int(re.sub(r"\D", "", item.frame_id) or "0"))):
        merged_manifest_rows.append(
            {
                "image_uid": record.image_uid,
                "dataset_id": record.dataset_id,
                "source_export_split": record.source_export_split,
                "assigned_split": record.assigned_split,
                "clip_id": record.clip_id,
                "frame_id": record.frame_id,
                "image_path": str(record.source_image_path),
                "output_file_name": record.output_file_name,
                "has_annotation": int(record.has_annotation),
                "annotation_count": record.annotation_count,
                "corpus_role": record.corpus_role,
            }
        )
        if not record.has_annotation:
            continue

        split = record.assigned_split
        image_dst = cleaned_root / "images" / split / record.output_file_name
        label_dst = cleaned_root / "labels" / split / f"{Path(record.output_file_name).stem}.txt"
        copy_image(record.source_image_path, image_dst)
        ensure_dir(label_dst.parent)

        coco_image = {
            "id": next_image_id,
            "license": 1,
            "file_name": record.output_file_name,
            "height": record.height,
            "width": record.width,
        }
        coco_by_split[split]["images"].append(coco_image)

        yolo_lines: list[str] = []
        for ann in anns_by_image[record.image_uid]:
            bbox = coerce_bbox(ann["bbox"])
            x_c, y_c, width_n, height_n = normalized_annotation_bbox(bbox, record.width, record.height)
            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {width_n:.6f} {height_n:.6f}")
            coco_by_split[split]["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                    "segmentation": [],
                }
            )
            annotation_metadata_rows.append(
                {
                    "annotation_id": f"seed_{next_ann_id}",
                    "image_uid": record.image_uid,
                    "dataset_id": record.dataset_id,
                    "clip_id": record.clip_id,
                    "frame_id": record.frame_id,
                    "image_path": str(record.source_image_path),
                    "class_name": "streetlight",
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "annotation_origin": "manual_seed_import",
                    "detector_version": "",
                    "reliability_version": "",
                    "detector_confidence": "",
                    "reliability_score": "",
                    "acceptance_band": "seed_manual",
                    "review_status": "imported_unverified",
                    "reviewer_id": "",
                    "review_timestamp": "",
                }
            )
            next_ann_id += 1

        label_dst.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        next_image_id += 1

    for split, coco in coco_by_split.items():
        split_dir = OUTPUT_ROOT / "cleaned_coco" / "merged" / split
        ensure_dir(split_dir)
        write_json(split_dir / "_annotations.coco.json", coco)

    dataset_yaml = "\n".join(
        [
            f"path: {cleaned_root.as_posix()}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            "",
            "names:",
            "  0: streetlight",
            "",
        ]
    )
    (cleaned_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    return cleaned_root, merged_manifest_rows, annotation_metadata_rows


def build_source_coco(records: list[ImageRecord], annotations: list[dict], dataset_id: str) -> dict:
    by_uid = {record.image_uid: record for record in records if record.dataset_id == dataset_id}
    anns_by_uid: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        if ann["dataset_id"] == dataset_id:
            anns_by_uid[ann["image_uid"]].append(ann)
    output = {
        "info": {"description": f"Cleaned {dataset_id} corpus", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "streetlight", "supercategory": "light_source"}],
    }
    next_image_id = 1
    next_ann_id = 1
    for image_uid, record in sorted(by_uid.items()):
        if not record.has_annotation:
            continue
        output["images"].append(
            {
                "id": next_image_id,
                "license": 1,
                "file_name": record.output_file_name,
                "height": record.height,
                "width": record.width,
            }
        )
        for ann in anns_by_uid[image_uid]:
            bbox = coerce_bbox(ann["bbox"])
            output["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                    "segmentation": [],
                }
            )
            next_ann_id += 1
        next_image_id += 1
    return output


def gather_local_review_candidates() -> list[dict]:
    candidates: list[dict] = []

    arindam_root = DATASETS_ROOT / "annotated_seed" / "arindam-annotated-images" / "nighttime-streetlight-detection.coco" / "train"
    arindam_coco = load_json(arindam_root / "_annotations.coco.json")
    annotated_ids = {ann["image_id"] for ann in arindam_coco["annotations"]}
    for image in arindam_coco["images"]:
        if image["id"] in annotated_ids:
            continue
        clip_id, frame_id = parse_clip_and_frame(image["file_name"])
        candidates.append(
            {
                "review_candidate_id": f"neg_seed_{len(candidates) + 1}",
                "source_pool": "arindam_unannotated_seed",
                "dataset_id": "arindam",
                "clip_id": clip_id,
                "frame_id": frame_id,
                "image_path": str(arindam_root / image["file_name"]),
                "priority": "high",
                "suggested_negative_subtype": "",
                "review_label": "pending",
                "notes": "Unannotated seed frame; verify clean negative vs missed positive.",
            }
        )

    extracted_items: list[dict] = []
    for image_path in sorted((DATASETS_ROOT / "extracted_frames").rglob("*.jpg")):
        clip_id = image_path.parent.name
        extracted_items.append(
            {
                "review_candidate_id": "",
                "source_pool": "local_extracted_night",
                "dataset_id": "local_mobile_night",
                "clip_id": clip_id,
                "frame_id": image_path.stem,
                "image_path": str(image_path),
                "priority": "medium",
                "suggested_negative_subtype": "",
                "review_label": "pending",
                "notes": "Sampled local extracted frame for hard-negative review.",
            }
        )

    hf_items: list[dict] = []
    hf_root = DATASETS_ROOT / "imported" / "huggingface" / "thirdeyelabs_indian_road_dataset" / "night_jpg"
    if hf_root.exists():
        for image_path in sorted(hf_root.rglob("*.jpg")):
            hf_items.append(
                {
                    "review_candidate_id": "",
                    "source_pool": "hf_night_external",
                    "dataset_id": "thirdeyelabs_night",
                    "clip_id": image_path.parent.name,
                    "frame_id": image_path.stem,
                    "image_path": str(image_path),
                    "priority": "medium",
                    "suggested_negative_subtype": "",
                    "review_label": "pending",
                    "notes": "Sampled external night frame for hard-negative review.",
                }
            )

    sampled_extracted = sample_evenly(extracted_items, 96)
    sampled_hf = sample_evenly(hf_items, 144)
    combined = candidates + sampled_extracted + sampled_hf
    for index, row in enumerate(combined, start=1):
        row["review_candidate_id"] = f"neg_{index:04d}"
    return combined


def build_calibration_subset(records: list[ImageRecord], annotations: list[dict]) -> list[dict]:
    by_uid = {record.image_uid: record for record in records}
    anns_by_uid: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_uid[ann["image_uid"]].append(ann)

    negative_candidates = [record for record in records if not record.has_annotation and record.dataset_id == "arindam"]
    multi_box: list[ImageRecord] = []
    single_small: list[ImageRecord] = []
    single_large: list[ImageRecord] = []

    for record in records:
        if not record.has_annotation:
            continue
        area_ratios = []
        for ann in anns_by_uid[record.image_uid]:
            bbox = coerce_bbox(ann["bbox"])
            area_ratios.append((bbox[2] * bbox[3]) / float(record.width * record.height))
        if record.annotation_count >= 2:
            multi_box.append(record)
        elif area_ratios and max(area_ratios) < 0.02:
            single_small.append(record)
        else:
            single_large.append(record)

    sampled_negatives = sample_evenly(
        [
            {
                "dataset_id": item.dataset_id,
                "clip_id": item.clip_id,
                "frame_id": item.frame_id,
                "image_path": str(item.source_image_path),
                "stratum": "negative_only",
            }
            for item in negative_candidates
        ],
        25,
    )
    sampled_multi = sample_evenly(
        [
            {
                "dataset_id": item.dataset_id,
                "clip_id": item.clip_id,
                "frame_id": item.frame_id,
                "image_path": str(item.source_image_path),
                "stratum": "multi_box",
            }
            for item in multi_box
        ],
        60,
    )
    sampled_small = sample_evenly(
        [
            {
                "dataset_id": item.dataset_id,
                "clip_id": item.clip_id,
                "frame_id": item.frame_id,
                "image_path": str(item.source_image_path),
                "stratum": "single_small",
            }
            for item in single_small
        ],
        60,
    )
    sampled_large = sample_evenly(
        [
            {
                "dataset_id": item.dataset_id,
                "clip_id": item.clip_id,
                "frame_id": item.frame_id,
                "image_path": str(item.source_image_path),
                "stratum": "single_large",
            }
            for item in single_large
        ],
        55,
    )

    combined = sampled_negatives + sampled_multi + sampled_small + sampled_large
    rows = []
    for index, item in enumerate(combined, start=1):
        rows.append(
            {
                "calibration_id": f"cal_{index:04d}",
                "dataset_id": item["dataset_id"],
                "clip_id": item["clip_id"],
                "frame_id": item["frame_id"],
                "image_path": item["image_path"],
                "stratum": item["stratum"],
                "review_status": "pending_manual_lock",
                "locked": 0,
                "notes": "Review and lock before using for reliability calibration.",
            }
        )
    return rows


def write_source_manifests(records: list[ImageRecord]) -> None:
    fieldnames = [
        "image_uid",
        "dataset_id",
        "source_export_split",
        "assigned_split",
        "clip_id",
        "frame_id",
        "image_path",
        "output_file_name",
        "has_annotation",
        "annotation_count",
        "corpus_role",
    ]
    manifest_root = OUTPUT_ROOT / "manifests"
    for dataset_id in ("jobin", "arindam"):
        rows = [
            {
                "image_uid": record.image_uid,
                "dataset_id": record.dataset_id,
                "source_export_split": record.source_export_split,
                "assigned_split": record.assigned_split,
                "clip_id": record.clip_id,
                "frame_id": record.frame_id,
                "image_path": str(record.source_image_path),
                "output_file_name": record.output_file_name,
                "has_annotation": int(record.has_annotation),
                "annotation_count": record.annotation_count,
                "corpus_role": record.corpus_role,
            }
            for record in records
            if record.dataset_id == dataset_id
        ]
        write_csv(manifest_root / f"{dataset_id}_manifest.csv", rows, fieldnames)


def write_remote_training_package(cleaned_root: Path) -> None:
    training_root = OUTPUT_ROOT / "training" / "remote_package"
    ensure_dir(training_root)
    dataset_yaml_src = cleaned_root / "dataset.yaml"
    shutil.copy2(dataset_yaml_src, training_root / "dataset.yaml")

    train_script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "$repoRoot = Split-Path -Parent $PSScriptRoot",
            "$datasetYaml = Join-Path $PSScriptRoot 'dataset.yaml'",
            "$modelPath = 'F:\\RBCCPS_Directory\\datasets\\yolov26_weights\\yolo26m.pt'",
            "$projectDir = Join-Path $repoRoot 'runs'",
            "",
            "python train_yolov26.py `",
            "  --model $modelPath `",
            "  --data $datasetYaml `",
            "  --imgsz 1280 `",
            "  --epochs 100 `",
            "  --batch 16 `",
            "  --device 0 `",
            "  --project $projectDir `",
            "  --name streetlight_detector_v1",
            "",
        ]
    )
    (training_root / "run_remote_training.ps1").write_text(train_script, encoding="utf-8")
    (training_root / "requirements-remote.txt").write_text("ultralytics>=8.3.222\n", encoding="utf-8")


def write_report(records: list[ImageRecord], merged_manifest_rows: list[dict], hard_negative_rows: list[dict], calibration_rows: list[dict]) -> None:
    report_root = OUTPUT_ROOT / "reports"
    ensure_dir(report_root)

    split_counts = Counter((row["dataset_id"], row["assigned_split"], row["corpus_role"]) for row in merged_manifest_rows)
    lines = [
        "# Annotation Automation Corpus Summary",
        "",
        "## Sources",
        "",
        "- `jobin` and `arindam` loaded from annotated seed exports",
        "- clip-level splits rebuilt locally from canonical frame names",
        "- class schema normalized to one `streetlight` class",
        "",
        "## Split Summary",
        "",
        "| Dataset | Split | Role | Images |",
        "| --- | --- | --- | ---: |",
    ]
    for (dataset_id, split_name, role), count in sorted(split_counts.items()):
        lines.append(f"| {dataset_id} | {split_name} | {role} | {count} |")

    lines.extend(
        [
            "",
            "## Review Manifests",
            "",
            f"- Hard-negative review manifest rows: `{len(hard_negative_rows)}`",
            f"- Calibration subset rows: `{len(calibration_rows)}`",
            "",
            "## Notes",
            "",
            "- Unannotated Arindam images are excluded from the cleaned training corpus until manual review.",
            "- Source exports are preserved unchanged; all outputs here are derived artifacts.",
            "",
        ]
    )
    (report_root / "corpus_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(OUTPUT_ROOT / "manifests")
    ensure_dir(OUTPUT_ROOT / "cleaned_coco")
    ensure_dir(OUTPUT_ROOT / "reviews")
    ensure_dir(OUTPUT_ROOT / "reports")

    jobin_records, jobin_annotations = load_jobin_source()
    arindam_records, arindam_annotations = load_arindam_source()
    all_records = jobin_records + arindam_records
    all_annotations = jobin_annotations + arindam_annotations

    split_map = assign_clip_splits(all_records)
    for record in all_records:
        record.assigned_split = split_map[f"{record.dataset_id}:{record.clip_id}"]

    write_source_manifests(all_records)

    cleaned_root, merged_manifest_rows, annotation_metadata_rows = build_cleaned_outputs(all_records, all_annotations)

    cleaned_coco_root = OUTPUT_ROOT / "cleaned_coco"
    write_json(cleaned_coco_root / "jobin_cleaned.coco.json", build_source_coco(all_records, all_annotations, "jobin"))
    write_json(cleaned_coco_root / "arindam_cleaned.coco.json", build_source_coco(all_records, all_annotations, "arindam"))

    merged_manifest_fieldnames = [
        "image_uid",
        "dataset_id",
        "source_export_split",
        "assigned_split",
        "clip_id",
        "frame_id",
        "image_path",
        "output_file_name",
        "has_annotation",
        "annotation_count",
        "corpus_role",
    ]
    write_csv(OUTPUT_ROOT / "manifests" / "merged_manifest.csv", merged_manifest_rows, merged_manifest_fieldnames)

    annotation_metadata_fieldnames = [
        "annotation_id",
        "image_uid",
        "dataset_id",
        "clip_id",
        "frame_id",
        "image_path",
        "class_name",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "annotation_origin",
        "detector_version",
        "reliability_version",
        "detector_confidence",
        "reliability_score",
        "acceptance_band",
        "review_status",
        "reviewer_id",
        "review_timestamp",
    ]
    write_csv(OUTPUT_ROOT / "manifests" / "annotation_metadata_seed.csv", annotation_metadata_rows, annotation_metadata_fieldnames)

    hard_negative_rows = gather_local_review_candidates()
    hard_negative_fieldnames = [
        "review_candidate_id",
        "source_pool",
        "dataset_id",
        "clip_id",
        "frame_id",
        "image_path",
        "priority",
        "suggested_negative_subtype",
        "review_label",
        "notes",
    ]
    write_csv(OUTPUT_ROOT / "reviews" / "hard_negative_review_manifest.csv", hard_negative_rows, hard_negative_fieldnames)

    calibration_rows = build_calibration_subset(all_records, all_annotations)
    calibration_fieldnames = [
        "calibration_id",
        "dataset_id",
        "clip_id",
        "frame_id",
        "image_path",
        "stratum",
        "review_status",
        "locked",
        "notes",
    ]
    write_csv(OUTPUT_ROOT / "reviews" / "calibration_subset_manifest.csv", calibration_rows, calibration_fieldnames)

    write_remote_training_package(cleaned_root)
    write_report(all_records, merged_manifest_rows, hard_negative_rows, calibration_rows)

    print(f"Processed annotation automation artifacts written to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
