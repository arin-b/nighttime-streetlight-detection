from __future__ import annotations

import csv
import json
import math
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "exports" / "chatgpt_annotation_batch_100"
STAGE = OUT_ROOT / "chatgpt_streetlight_measurement_batch_100"
ZIP_PATH = OUT_ROOT / "chatgpt_streetlight_measurement_batch_100.zip"

MOBILE_ROOT = ROOT / "datasets" / "extracted_frames" / "mobile_night_videos"
SEED_ROOT = ROOT / "datasets" / "annotated_seed" / "jobin-original-annotated-images"
WORKFLOW_MD = ROOT / "measurement_llm_annotation_workflow.md"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


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
        if source_pool == "mobile_night_video":
            source_group = path.parent.name
        else:
            source_group = path.parent.name
        rows.append(Candidate(path, source_pool, source_group, width, height, brightness, contrast, sharpness))
    return rows


def spread_sample(candidates: list[Candidate], count: int, seed: int) -> list[Candidate]:
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return list(candidates)
    rng = random.Random(seed)
    groups: dict[str, list[Candidate]] = {}
    for item in candidates:
        groups.setdefault(item.source_group, []).append(item)

    selected: list[Candidate] = []
    sorted_groups = sorted(groups.items(), key=lambda row: (-len(row[1]), row[0]))
    base = max(1, count // max(1, len(sorted_groups)))
    remainder = count

    for group_index, (_group, rows) in enumerate(sorted_groups):
        if remainder <= 0:
            break
        remaining_groups = len(sorted_groups) - group_index
        target = min(len(rows), max(1, min(base + 2, math.ceil(remainder / remaining_groups))))
        rows = sorted(rows, key=lambda r: (r.brightness, r.contrast, r.sharpness, str(r.path)))
        picks = quantile_pick(rows, target)
        selected.extend(picks)
        remainder = count - len(selected)

    if len(selected) < count:
        used = {item.path for item in selected}
        leftover = [item for item in candidates if item.path not in used]
        rng.shuffle(leftover)
        selected.extend(leftover[: count - len(selected)])

    return selected[:count]


def quantile_pick(rows: list[Candidate], count: int) -> list[Candidate]:
    if count >= len(rows):
        return list(rows)
    if count == 1:
        return [rows[len(rows) // 2]]
    picks: list[Candidate] = []
    used: set[int] = set()
    for i in range(count):
        index = round(i * (len(rows) - 1) / (count - 1))
        while index in used and index + 1 < len(rows):
            index += 1
        if index in used:
            index = next(j for j in range(len(rows)) if j not in used)
        used.add(index)
        picks.append(rows[index])
    return picks


def make_prompt() -> str:
    return """# Prompt To Paste Into ChatGPT

You are a careful visual annotation agent creating draft labels for a streetlight measurement dataset.

I uploaded a zip containing:

- `images/`: 100 images.
- `image_manifest.csv`: explicit image dimensions and original source metadata.
- `image_manifest.json`: same metadata in JSON.
- `measurement_llm_annotation_workflow.md`: the annotation workflow and QA rules.

Annotate the images one by one using the workflow. Return structured JSON only. Use pixel coordinates in each image's native dimensions from the manifest. Do not infer or invent calibrated lux values.

These are draft labels for human verification, so every image must include:

```json
{"track_id": "", "flag": "needs_human_verification"}
```

For every image, output this schema:

```json
{
  "image_name": "",
  "width": 0,
  "height": 0,
  "review_status": "needs_review",
  "boxes": [
    {
      "box_id": "box_001",
      "class_name": "streetlight_lamp_head",
      "bbox_xyxy": [0, 0, 0, 0],
      "track_id": "track_001",
      "parent_pole_box_id": "",
      "status": "fixed",
      "source": "chatgpt_visual_draft",
      "confidence": "high",
      "notes": ""
    }
  ],
  "confounder_boxes": [
    {
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
    }
  ],
  "polygons": [
    {
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
    }
  ],
  "measurement": {
    "lamp_status": [
      {"track_id": "track_001", "status": "on"}
    ],
    "public_space_regions": [
      {
        "region_type": "road",
        "points": [[0, 0], [0, 0], [0, 0]],
        "confidence": "medium",
        "source": "chatgpt_visual_draft"
      }
    ],
    "affected_regions": [
      {
        "track_id": "track_001",
        "region_type": "lit_area",
        "visibility_quality": "visible",
        "points": [[0, 0], [0, 0], [0, 0]],
        "confidence": "medium",
        "source": "chatgpt_visual_draft",
        "notes": ""
      }
    ],
    "visibility_labels": [
      {"track_id": "track_001", "visibility_class": "adequate"}
    ],
    "attribution_labels": [
      {"track_id": "track_001", "attribution_class": "mixed", "evidence": "visual draft; verify"}
    ],
    "lux_points": [],
    "qa_flags": [
      {"track_id": "", "flag": "needs_human_verification"}
    ]
  }
}
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


def write_outputs(selected: list[Candidate]) -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (STAGE / "images").mkdir(parents=True, exist_ok=True)

    rows = []
    for index, item in enumerate(selected, start=1):
        ext = item.path.suffix.lower()
        name = f"{index:03d}_{item.source_pool}_{item.source_group}_{item.path.stem}{ext}"
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        dst = STAGE / "images" / safe_name
        shutil.copy2(item.path, dst)
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

    fieldnames = list(rows[0].keys())
    with (STAGE / "image_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (STAGE / "image_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    shutil.copy2(WORKFLOW_MD, STAGE / "measurement_llm_annotation_workflow.md")
    (STAGE / "PROMPT_TO_PASTE_INTO_CHATGPT.md").write_text(make_prompt(), encoding="utf-8")
    (STAGE / "README.md").write_text(
        "# ChatGPT Streetlight Measurement Draft Annotation Batch\n\n"
        "Upload this zip to ChatGPT. Paste the prompt from `PROMPT_TO_PASTE_INTO_CHATGPT.md`.\n\n"
        "The manifest gives explicit dimensions for every image. ChatGPT should return JSON draft labels only.\n",
        encoding="utf-8",
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(STAGE.rglob("*")):
            archive.write(path, path.relative_to(STAGE.parent))


def main() -> None:
    mobile = collect_images(MOBILE_ROOT, "mobile_night_video")
    seed = collect_images(SEED_ROOT, "seed_streetlight_dataset")
    selected_mobile = spread_sample(mobile, 70, seed=20260528)
    selected_seed = spread_sample(seed, 30, seed=20260529)
    selected = selected_mobile + selected_seed
    selected = selected[:100]
    if len(selected) < 100:
        raise SystemExit(f"Only found {len(selected)} usable images")
    write_outputs(selected)
    print(json.dumps({
        "zip": str(ZIP_PATH),
        "stage": str(STAGE),
        "count": len(selected),
        "mobile": len(selected_mobile),
        "seed": len(selected_seed),
        "zip_size_mb": round(ZIP_PATH.stat().st_size / (1024 * 1024), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
