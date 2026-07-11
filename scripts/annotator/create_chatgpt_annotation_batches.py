from __future__ import annotations

import csv
import json
import math
import random
import shutil
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "exports" / "chatgpt batches"
WORKFLOW_MD = ROOT / "measurement_llm_annotation_workflow.md"

MOBILE_ROOT = ROOT / "datasets" / "extracted_frames" / "mobile_night_videos"
SEED_ROOT = ROOT / "datasets" / "annotated_seed"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 100


@dataclass
class Candidate:
    path: Path
    source_pool: str
    source_group: str
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float


def image_stats(path: Path) -> tuple[int, int, float, float, float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        gray = image.convert("L").resize((160, max(1, round(160 * height / width))))
        stat = ImageStat.Stat(gray)
        brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = float(ImageStat.Stat(edges).mean[0])
        return width, height, brightness, contrast, sharpness


def collect_images(root: Path, source_pool: str) -> list[Candidate]:
    rows: list[Candidate] = []
    if not root.exists():
        return rows
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        try:
            width, height, brightness, contrast, sharpness = image_stats(path)
        except Exception:
            continue
        rows.append(Candidate(path, source_pool, path.parent.name, width, height, brightness, contrast, sharpness))
    return rows


def interleave_groups(candidates: list[Candidate]) -> list[Candidate]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for item in candidates:
        groups[item.source_group].append(item)

    ordered_groups = sorted(groups.items(), key=lambda row: (-len(row[1]), row[0]))
    queues = {
        group: deque(sorted(rows, key=lambda r: (r.brightness, r.contrast, r.sharpness, str(r.path))))
        for group, rows in ordered_groups
    }

    ordered: list[Candidate] = []
    while any(queues.values()):
        for group, _rows in ordered_groups:
            queue = queues[group]
            if queue:
                ordered.append(queue.popleft())
    return ordered


def make_prompt(image_count: int) -> str:
    return f"""# Prompt To Paste Into ChatGPT

You are a careful visual annotation agent creating draft labels for a streetlight measurement dataset.

I uploaded a zip containing:

- `images/`: {image_count} images.
- `image_manifest.csv`: explicit image dimensions and original source metadata.
- `image_manifest.json`: same metadata in JSON.
- `measurement_llm_annotation_workflow.md`: the annotation workflow and QA rules.

Annotate the images one by one using the workflow. Return structured JSON only. Use pixel coordinates in each image's native dimensions from the manifest. Do not infer or invent calibrated lux values.

These are draft labels for human verification, so every image must include:

```json
{{"track_id": "", "flag": "needs_human_verification"}}
```

For every image, output this schema:

```json
{{
  "image_name": "",
  "width": 0,
  "height": 0,
  "review_status": "needs_review",
  "boxes": [
    {{
      "box_id": "box_001",
      "class_name": "streetlight_lamp_head",
      "bbox_xyxy": [0, 0, 0, 0],
      "track_id": "track_001",
      "parent_pole_box_id": "",
      "status": "fixed",
      "source": "chatgpt_visual_draft",
      "confidence": "high",
      "notes": ""
    }}
  ],
  "confounder_boxes": [
    {{
      "box_id": "other_box_001",
      "surface_type": "shopfront",
      "bbox_xyxy": [0, 0, 0, 0],
      "source": "chatgpt_visual_draft",
      "confidence": "medium",
      "is_bright_source": true,
      "is_reflective": false,
      "is_public_space": false,
      "can_confound_streetlight": true,
      "overlaps_affected_region": false,
      "augmentation_allowed": false,
      "notes": ""
    }}
  ],
  "polygons": [
    {{
      "polygon_id": "poly_001",
      "surface_type": "wet_road_reflection",
      "points": [[0, 0], [0, 0], [0, 0]],
      "source": "chatgpt_visual_draft",
      "confidence": "medium",
      "is_bright_source": false,
      "is_reflective": true,
      "is_public_space": true,
      "can_confound_streetlight": true,
      "overlaps_affected_region": true,
      "augmentation_allowed": false,
      "notes": ""
    }}
  ],
  "measurement": {{
    "lamp_status": [
      {{"track_id": "track_001", "status": "on"}}
    ],
    "public_space_regions": [
      {{
        "region_type": "road",
        "points": [[0, 0], [0, 0], [0, 0]],
        "confidence": "medium",
        "source": "chatgpt_visual_draft"
      }}
    ],
    "affected_regions": [
      {{
        "track_id": "track_001",
        "region_type": "lit_area",
        "visibility_quality": "visible",
        "points": [[0, 0], [0, 0], [0, 0]],
        "confidence": "medium",
        "source": "chatgpt_visual_draft",
        "notes": ""
      }}
    ],
    "visibility_labels": [
      {{"track_id": "track_001", "visibility_class": "adequate"}}
    ],
    "attribution_labels": [
      {{"track_id": "track_001", "attribution_class": "mixed", "evidence": "visual draft; verify"}}
    ],
    "lux_points": [],
    "qa_flags": [
      {{"track_id": "", "flag": "needs_human_verification"}}
    ]
  }}
}}
```

Rules:

- Draw tight `streetlight_lamp_head` boxes around lamp heads/light sources.
- Draw `streetlight_pole` boxes only when the pole or fixture support is visible enough.
- If pole is not visible, add QA flag `pole_not_visible`.
- Mark major confounders: shopfronts, signs, bright windows, headlights, wet reflections, unknown bright sources.
- Use polygons for road/footpath and lit area. Keep polygons detailed but do not waste effort on sub-pixel perfection.
- Lit area is per selected lamp/track. If attribution is unclear, set `visibility_quality` to `partly_visible` or `not_visible` and add uncertainty flags.
- Rate visibility per lamp/track, not globally.
- Use `unknown` instead of guessing when unclear.
- Do not output prose explanations. Output JSON array chunks of 5-10 images at a time.
"""


def write_manifest_and_assets(stage: Path, selected: list[Candidate]) -> None:
    (stage / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(selected, start=1):
        ext = item.path.suffix.lower()
        name = f"{index:03d}_{item.source_pool}_{item.source_group}_{item.path.stem}{ext}"
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        shutil.copy2(item.path, stage / "images" / safe_name)
        rows.append(
            {
                "batch_index": index,
                "image_name": safe_name,
                "width": item.width,
                "height": item.height,
                "source_pool": item.source_pool,
                "source_group": item.source_group,
                "original_path": str(item.path),
                "brightness_mean": round(item.brightness, 3),
                "contrast_std": round(item.contrast, 3),
                "edge_mean": round(item.sharpness, 3),
            }
        )

    with (stage / "image_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (stage / "image_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    shutil.copy2(WORKFLOW_MD, stage / "measurement_llm_annotation_workflow.md")
    (stage / "PROMPT_TO_PASTE_INTO_CHATGPT.md").write_text(make_prompt(len(rows)), encoding="utf-8")
    (stage / "README.md").write_text(
        "# ChatGPT Streetlight Measurement Draft Annotation Batch\n\n"
        "Upload this zip to ChatGPT. Paste the prompt from `PROMPT_TO_PASTE_INTO_CHATGPT.md`.\n\n"
        "The manifest gives explicit dimensions for every image. ChatGPT should return JSON draft labels only.\n",
        encoding="utf-8",
    )


def zip_stage(stage: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob("*")):
            archive.write(path, path.relative_to(stage.parent))


def partition_batches(mobile: list[Candidate], seed: list[Candidate]) -> list[list[Candidate]]:
    total = len(mobile) + len(seed)
    if total == 0:
        return []
    batch_size = BATCH_SIZE
    mobile_per_batch = max(1, round(batch_size * len(mobile) / total))
    seed_per_batch = batch_size - mobile_per_batch

    mobile_ordered = interleave_groups(mobile)
    seed_ordered = interleave_groups(seed)

    batches: list[list[Candidate]] = []
    mobile_index = 0
    seed_index = 0
    while mobile_index < len(mobile_ordered) or seed_index < len(seed_ordered):
        batch_mobile = mobile_ordered[mobile_index: mobile_index + mobile_per_batch]
        batch_seed = seed_ordered[seed_index: seed_index + seed_per_batch]
        mobile_index += len(batch_mobile)
        seed_index += len(batch_seed)
        batch = [item for pair in zip_longest(batch_mobile, batch_seed) for item in pair if item is not None]
        batches.append(batch)
    return batches


def main() -> None:
    mobile = collect_images(MOBILE_ROOT, "mobile_night_video")
    seed = collect_images(SEED_ROOT, "seed_streetlight_dataset")
    all_batches = partition_batches(mobile, seed)
    if not all_batches:
        raise SystemExit("No images found to package.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for stale_zip in OUT_ROOT.glob("chatgpt_streetlight_measurement_batch_*.zip"):
        stale_zip.unlink()
    summary_rows = []

    for batch_index, selected in enumerate(all_batches, start=1):
        batch_name = f"chatgpt_streetlight_measurement_batch_{batch_index:03d}_{len(selected):03d}"
        stage = OUT_ROOT / batch_name
        zip_path = OUT_ROOT / f"{batch_name}.zip"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=True)

        write_manifest_and_assets(stage, selected)
        zip_stage(stage, zip_path)
        shutil.rmtree(stage)

        summary_rows.append(
            {
                "batch_index": batch_index,
                "image_count": len(selected),
                "zip_name": zip_path.name,
                "zip_path": str(zip_path),
                "mobile_count": sum(1 for item in selected if item.source_pool == "mobile_night_video"),
                "seed_count": sum(1 for item in selected if item.source_pool == "seed_streetlight_dataset"),
            }
        )

    (OUT_ROOT / "batch_index.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    with (OUT_ROOT / "batch_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (OUT_ROOT / "README.md").write_text(
        "# ChatGPT Batches\n\n"
        "This folder contains self-contained annotation zips for the full local candidate pool.\n"
        f"Total images packaged: {len(mobile) + len(seed)}\n"
        f"Total batches: {len(all_batches)}\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "output_root": str(OUT_ROOT),
            "batches": len(all_batches),
            "images_total": len(mobile) + len(seed),
            "mobile": len(mobile),
            "seed": len(seed),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
