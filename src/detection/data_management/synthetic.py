"""Reproducible, label-preserving nighttime streetlight synthesis.

The module deliberately produces a *separate* training-only pool.  It never
opens an original file for writing and it refuses source scenes represented in
validation or test.  YOLO detection labels are copied unchanged because the
transformations change emitted light, not physical object geometry.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GENERATION_VERSION = "synthetic-illumination-v1"
DEFAULT_EXTRACTION_COMMAND = "ffmpeg -i video.mp4 fps=1 frames/frame\\_%06d.jpg"


@dataclass(frozen=True)
class YoloAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    raw_line: str

    def pixel_box(self, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
        width, height = image_size
        x1 = (self.x_center - self.width / 2) * width
        y1 = (self.y_center - self.height / 2) * height
        x2 = (self.x_center + self.width / 2) * width
        y2 = (self.y_center + self.height / 2) * height
        return max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)


@dataclass
class SourceRecord:
    image_path: Path
    annotation_path: Path
    source_id: str
    split: str
    source_scene_id: str
    source_video: str | None = None
    source_frame_index: int | None = None
    source_timestamp: float | None = None
    annotations: list[YoloAnnotation] = field(default_factory=list)
    exact_hash: str = ""
    perceptual_hash: int = 0
    confounder_score: float = 0.0
    confounder_category: str = "ordinary"


@dataclass
class SynthesisConfig:
    input_images: Path
    input_annotations: Path
    output_images: Path
    output_annotations: Path
    output_metadata: Path
    source_manifest: Path | None = None
    manual_scene_labels: Path | None = None
    temporal_exclusion_seconds: float = 1.0
    perceptual_duplicate_threshold: int = 8
    synthetic_fraction: float = 0.20
    number_of_source_frames: int | None = None
    confounder_fraction: float = 0.40
    ordinary_fraction: float = 0.60
    illumination_states: tuple[str, ...] = ("dim", "severe_dim", "flicker", "off", "mixed")
    state_probabilities: dict[str, float] = field(default_factory=lambda: {"dim": .25, "severe_dim": .25, "flicker": .20, "off": .20, "mixed": .10})
    dimming_severity_range: tuple[float, float] = (.55, .78)
    severe_dimming_severity_range: tuple[float, float] = (.12, .45)
    flicker_parameters: dict[str, Any] = field(default_factory=lambda: {"residual_range": [.08, .90]})
    off_residual: float = .08
    number_of_variants_per_source: int | tuple[int, int] = 2
    max_lights_per_variant: int = 2
    luminous_class_ids: tuple[int, ...] = ()
    target_class_ids: tuple[int, ...] = (0,)
    fallback_luminous_height_fraction: float = .25
    mask_blur_radius_fraction: float = .16
    confounder_brightness_threshold: int = 205
    confounder_score_threshold: float = .045
    random_seed: int = 20260912
    dry_run: bool = False
    overwrite: bool = False
    generate_preview: bool = False
    preview_limit: int = 20
    generation_version: str = GENERATION_VERSION
    extraction_command: str = DEFAULT_EXTRACTION_COMMAND


def _path(value: Any, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _state_name(value: Any) -> str:
    # YAML 1.1 treats unquoted `off` as boolean false.  Accept it gracefully
    # while retaining a normal string API for callers and metadata.
    return "off" if value is False else str(value)


def load_config(path: str | Path) -> SynthesisConfig:
    """Load a YAML/JSON config; relative paths are relative to its file."""
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    # Repository configs conventionally use project-root-relative dataset paths.
    # A standalone config (e.g. a test fixture) instead resolves beside itself.
    base = next((parent for parent in (config_path.parent, *config_path.parents) if (parent / "pyproject.toml").exists()), config_path.parent)
    required = ("input_images", "input_annotations", "output_images", "output_annotations", "output_metadata")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"Missing required synthetic-data configuration: {', '.join(missing)}")
    path_keys = {*required, "source_manifest", "manual_scene_labels"}
    normalized = {key: (_path(value, base) if key in path_keys and value else value) for key, value in payload.items()}
    if "illumination_states" in normalized:
        normalized["illumination_states"] = tuple(_state_name(item) for item in normalized["illumination_states"])
    if "state_probabilities" in normalized:
        normalized["state_probabilities"] = {_state_name(key): float(value) for key, value in normalized["state_probabilities"].items()}
    for key in ("luminous_class_ids", "target_class_ids"):
        if key in normalized:
            normalized[key] = tuple(normalized[key])
    for key in ("dimming_severity_range", "severe_dimming_severity_range"):
        if key in normalized:
            normalized[key] = tuple(float(item) for item in normalized[key])
    config = SynthesisConfig(**normalized)
    validate_config(config)
    return config


def validate_config(config: SynthesisConfig) -> None:
    if not config.input_images.is_dir() or not config.input_annotations.is_dir():
        raise ValueError("input_images and input_annotations must be existing directories")
    if not 0 <= config.confounder_fraction <= 1 or not 0 <= config.ordinary_fraction <= 1:
        raise ValueError("confounder_fraction and ordinary_fraction must be within [0, 1]")
    if not math.isclose(config.confounder_fraction + config.ordinary_fraction, 1.0, abs_tol=1e-6):
        raise ValueError("confounder_fraction + ordinary_fraction must equal 1")
    if not 0 < config.synthetic_fraction <= 1:
        raise ValueError("synthetic_fraction must be in (0, 1]")
    if config.number_of_source_frames is not None and config.number_of_source_frames < 1:
        raise ValueError("number_of_source_frames must be positive when provided")
    if set(config.state_probabilities) - set(config.illumination_states):
        raise ValueError("state_probabilities contains an unavailable illumination state")
    if not config.state_probabilities or sum(config.state_probabilities.values()) <= 0:
        raise ValueError("state_probabilities must have positive total weight")
    for name, bounds in (("dimming_severity_range", config.dimming_severity_range), ("severe_dimming_severity_range", config.severe_dimming_severity_range)):
        if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] <= 1:
            raise ValueError(f"{name} must be two residual illumination factors in [0, 1]")


def parse_yolo_annotations(path: Path) -> list[YoloAnnotation]:
    """Parse standard detection labels, preserving raw lines for exact copying."""
    if not path.exists():
        return []
    annotations = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        values = text.split()
        if len(values) < 5:
            raise ValueError(f"Invalid YOLO annotation {path}:{line_number}")
        try:
            annotation = YoloAnnotation(int(float(values[0])), *map(float, values[1:5]), raw)
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO annotation {path}:{line_number}") from exc
        if not all(0 <= item <= 1 for item in (annotation.x_center, annotation.y_center, annotation.width, annotation.height)):
            raise ValueError(f"Out-of-range YOLO annotation {path}:{line_number}")
        annotations.append(annotation)
    return annotations


def _read_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise ValueError(f"source_manifest does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        row = {str(key): "" if value is None else str(value) for key, value in row.items()}
        for key in ("source_image", "image_path", "image", "file_name", "filename"):
            if row.get(key):
                result[Path(row[key]).name] = row
    return result


def _normal_split(value: str) -> str:
    value = value.strip().lower()
    if value in {"train", "training"}:
        return "train"
    if value in {"valid", "val", "validation"}:
        return "valid"
    if value in {"test", "testing"}:
        return "test"
    return "unknown"


def _infer_split(image: Path, root: Path, manifest: dict[str, str]) -> str:
    for key in ("split", "assigned_split", "source_export_split"):
        if manifest.get(key):
            return _normal_split(manifest[key])
    path_tokens = {part.lower() for part in image.relative_to(root).parts}
    for token in path_tokens:
        result = _normal_split(token)
        if result != "unknown":
            return result
    # Flat legacy corpus names contain e.g. _images_train_jobin__...
    match = re.search(r"(?:^|_)(train|valid|val|test)(?:_|$)", image.stem.lower())
    return _normal_split(match.group(1)) if match else "unknown"


def _frame_index(stem: str, metadata: dict[str, str]) -> int | None:
    for key in ("source_frame_index", "frame_index", "frame_no", "frame"):
        if metadata.get(key, "").strip().isdigit():
            return int(metadata[key])
    match = re.search(r"_frame_(\d+)", stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _scene_id(stem: str, metadata: dict[str, str]) -> str:
    for key in ("source_scene_id", "scene_id", "clip_id", "source_video", "video_id"):
        if metadata.get(key, "").strip():
            return metadata[key].strip()
    # Explicitly a filename-derived grouping, never a claimed timestamp/video id.
    return re.sub(r"_frame_\d+.*$", "", stem, flags=re.IGNORECASE) or stem


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value not in {None, ""} else None
    except ValueError:
        return None


def load_source_records(config: SynthesisConfig) -> list[SourceRecord]:
    manifest = _read_manifest(config.source_manifest)
    records: list[SourceRecord] = []
    for image_path in sorted(path for path in config.input_images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        metadata = manifest.get(image_path.name, {})
        label_name = image_path.with_suffix(".txt").name
        candidates = [config.input_annotations / label_name]
        try:
            relative_parent = image_path.relative_to(config.input_images).parent
            candidates.insert(0, config.input_annotations / relative_parent / label_name)
        except ValueError:
            pass
        annotation_path = next((path for path in candidates if path.exists()), candidates[0])
        annotations = parse_yolo_annotations(annotation_path)
        timestamp = _float_or_none(metadata.get("source_timestamp") or metadata.get("timestamp"))
        record = SourceRecord(
            image_path=image_path,
            annotation_path=annotation_path,
            source_id=metadata.get("source_id") or image_path.stem,
            split=_infer_split(image_path, config.input_images, metadata),
            source_scene_id=_scene_id(image_path.stem, metadata),
            source_video=metadata.get("source_video") or metadata.get("video_id") or None,
            source_frame_index=_frame_index(image_path.stem, metadata),
            source_timestamp=timestamp,
            annotations=annotations,
        )
        records.append(record)
    return records


def exact_image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel_values(image: Image.Image) -> list[int]:
    # Pillow 12 renamed this accessor and emits a deprecation warning for
    # getdata(); keeping both branches supports the project's declared Pillow 10.
    if hasattr(image, "get_flattened_data"):
        return list(image.get_flattened_data())
    return list(image.getdata())


def perceptual_hash(image: Image.Image, size: int = 16) -> int:
    """A dependency-free average hash. Hamming distance is the similarity metric."""
    gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
    values = _pixel_values(gray)
    mean = sum(values) / len(values)
    result = 0
    for value in values:
        result = (result << 1) | int(value >= mean)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def has_eligible_streetlight(record: SourceRecord, config: SynthesisConfig) -> bool:
    return any(annotation.class_id in config.target_class_ids for annotation in record.annotations)


def _manual_categories(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise ValueError(f"manual_scene_labels does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = row.get("scene_id") or row.get("source_scene_id") or row.get("source_id")
        category = (row.get("category") or "").strip().lower()
        if key and category in {"ordinary", "confounder_heavy"}:
            result[key] = category
    return result


def confounder_score(image: Image.Image, annotations: Iterable[YoloAnnotation], threshold: int) -> float:
    """Heuristic brightness score outside labelled luminaires; not semantic truth."""
    small = ImageOps.grayscale(image).resize((160, max(1, round(image.height * 160 / image.width))), Image.Resampling.BILINEAR)
    mask = Image.new("L", small.size, 0)
    draw = ImageDraw.Draw(mask)
    scale_x, scale_y = small.width / image.width, small.height / image.height
    for annotation in annotations:
        x1, y1, x2, y2 = annotation.pixel_box(image.size)
        draw.rectangle((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y), fill=255)
    pixels, protected = _pixel_values(small), _pixel_values(mask)
    values = [pixel for pixel, excluded in zip(pixels, protected) if not excluded]
    if not values:
        return 0.0
    bright = sum(pixel >= threshold for pixel in values) / len(values)
    # Retain the upper-tail intensity as a small signal for glare/signage-heavy frames.
    tail = sum(values) / len(values) / 255.0
    return round(.75 * bright + .25 * max(0.0, tail - .25), 6)


def classify_scene_difficulty(records: list[SourceRecord], config: SynthesisConfig) -> None:
    overrides = _manual_categories(config.manual_scene_labels)
    for record in records:
        with Image.open(record.image_path) as image:
            score = confounder_score(image.convert("RGB"), record.annotations, config.confounder_brightness_threshold)
        record.confounder_score = score
        record.confounder_category = overrides.get(record.source_scene_id, "confounder_heavy" if score >= config.confounder_score_threshold else "ordinary")


def filter_candidates(records: list[SourceRecord], config: SynthesisConfig) -> tuple[list[SourceRecord], Counter[str]]:
    """Apply exact, temporal and visual filtering with per-scene bounded comparisons."""
    counts: Counter[str] = Counter()
    selected: list[SourceRecord] = []
    accepted_by_scene: dict[str, list[SourceRecord]] = defaultdict(list)
    exact_seen: set[str] = set()
    for record in sorted(records, key=lambda item: (item.source_scene_id, item.source_timestamp is None, item.source_timestamp or 0, item.source_frame_index or -1, item.image_path.name)):
        record.exact_hash = exact_image_hash(record.image_path)
        if record.exact_hash in exact_seen:
            counts["exact_duplicate"] += 1
            continue
        peers = accepted_by_scene[record.source_scene_id]
        if record.source_timestamp is not None and any(peer.source_timestamp is not None and abs(record.source_timestamp - peer.source_timestamp) < config.temporal_exclusion_seconds for peer in peers):
            counts["temporal_duplicate"] += 1
            continue
        with Image.open(record.image_path) as image:
            record.perceptual_hash = perceptual_hash(image)
        # A clip/source scene is a natural visual index and avoids global O(N²) matching.
        if any(hamming_distance(record.perceptual_hash, peer.perceptual_hash) <= config.perceptual_duplicate_threshold for peer in peers[-32:]):
            counts["perceptual_duplicate"] += 1
            continue
        exact_seen.add(record.exact_hash)
        peers.append(record)
        selected.append(record)
    return selected, counts


def select_source_frames(records: list[SourceRecord], config: SynthesisConfig) -> tuple[list[SourceRecord], list[str]]:
    target = config.number_of_source_frames or max(1, round(len(records) * config.synthetic_fraction))
    target = min(target, len(records))
    confounded = [record for record in records if record.confounder_category == "confounder_heavy"]
    ordinary = [record for record in records if record.confounder_category == "ordinary"]
    wanted_confounded = round(target * config.confounder_fraction)
    warnings_out: list[str] = []
    if len(confounded) < wanted_confounded:
        warnings_out.append(f"Insufficient confounder-heavy candidates: requested {wanted_confounded}, available {len(confounded)}; 40% target cannot be achieved.")
    rng = random.Random(config.random_seed)
    rng.shuffle(confounded)
    rng.shuffle(ordinary)
    chosen_confounded = confounded[:wanted_confounded]
    chosen_ordinary = ordinary[: target - len(chosen_confounded)]
    if len(chosen_ordinary) < target - len(chosen_confounded):
        remaining = [item for item in confounded[len(chosen_confounded):] if item not in chosen_confounded]
        chosen_confounded.extend(remaining[: target - len(chosen_confounded) - len(chosen_ordinary)])
    selected = chosen_confounded + chosen_ordinary
    rng.shuffle(selected)
    return selected, warnings_out


def luminous_region(annotation: YoloAnnotation, image_size: tuple[int, int], config: SynthesisConfig) -> tuple[float, float, float, float]:
    """Use a dedicated halo class where configured, otherwise conservative top-of-box fallback."""
    x1, y1, x2, y2 = annotation.pixel_box(image_size)
    if annotation.class_id in config.luminous_class_ids:
        return x1, y1, x2, y2
    height = max(1.0, y2 - y1)
    luminous_height = max(2.0, height * config.fallback_luminous_height_fraction)
    # Full-streetlight boxes may be pole-heavy; only their upper fixture area is changed.
    return x1, y1, x2, min(y2, y1 + luminous_height)


def create_soft_illumination_mask(image_size: tuple[int, int], region: tuple[float, float, float, float], blur_fraction: float = .16) -> Image.Image:
    x1, y1, x2, y2 = region
    width, height = image_size
    region_width, region_height = max(2.0, x2 - x1), max(2.0, y2 - y1)
    # An ellipse avoids a rectangular visual signature; blur models a soft halo falloff.
    pad_x, pad_y = region_width * .18, region_height * .35
    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).ellipse((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), fill=255)
    radius = max(1.0, min(width, height) * blur_fraction * max(region_width / width, region_height / height))
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def combine_masks(masks: list[Image.Image], size: tuple[int, int]) -> Image.Image:
    result = Image.new("L", size, 0)
    for mask in masks:
        result = ImageChops.lighter(result, mask)
    return result


def apply_dim(image: Image.Image, mask: Image.Image, residual: float) -> Image.Image:
    if not 0 <= residual <= 1:
        raise ValueError("residual illumination must be in [0, 1]")
    # LUT multiplication preserves colour balance and Image.composite gives soft alpha(x).
    lut = [round(value * residual) for value in range(256)]
    dimmed = image.point(lut * len(image.getbands()))
    return Image.composite(dimmed, image, mask)


def apply_off(image: Image.Image, mask: Image.Image, residual: float = .08) -> Image.Image:
    return apply_dim(image, mask, residual)


def _variant_rng(seed: int, source_id: str, variant_index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{source_id}|{variant_index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _sample_state(rng: random.Random, weights: dict[str, float]) -> str:
    states, values = zip(*sorted(weights.items()))
    return rng.choices(states, weights=values, k=1)[0]


def _variant_count(config: SynthesisConfig, rng: random.Random) -> int:
    if isinstance(config.number_of_variants_per_source, int):
        return config.number_of_variants_per_source
    low, high = config.number_of_variants_per_source
    return rng.randint(int(low), int(high))


def _eligible_annotations(record: SourceRecord, config: SynthesisConfig) -> list[tuple[int, YoloAnnotation]]:
    return [(index, annotation) for index, annotation in enumerate(record.annotations) if annotation.class_id in config.target_class_ids]


def _sample_residual(state: str, rng: random.Random, config: SynthesisConfig) -> float:
    if state == "dim":
        return rng.uniform(*config.dimming_severity_range)
    if state == "severe_dim":
        return rng.uniform(*config.severe_dimming_severity_range)
    if state == "flicker":
        low, high = config.flicker_parameters.get("residual_range", [.08, .90])
        return rng.uniform(float(low), float(high))
    if state == "off":
        return config.off_residual
    raise ValueError(f"No single residual for state {state}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._") or "source"


def _write_preview(original: Image.Image, mask: Image.Image, synthetic: Image.Image, destination: Path) -> None:
    panels = [original.convert("RGB"), Image.merge("RGB", (mask, mask, mask)), synthetic.convert("RGB")]
    labels = ("ORIGINAL", "SYNTHESIS MASK", "SYNTHETIC")
    label_height = 28
    sheet = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels) + label_height), "black")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for panel, label in zip(panels, labels):
        sheet.paste(panel, (x, label_height))
        draw.text((x + 6, 6), label, fill="white")
        x += panel.width
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def _prepare_outputs(config: SynthesisConfig) -> None:
    # Metadata commonly lives beside image/label directories, so its parent is
    # intentionally allowed to contain those directories.  Only an actual
    # output target is protected from an accidental rerun.
    for path in (config.output_images, config.output_annotations):
        if path.exists() and any(path.iterdir()) and not config.overwrite:
            raise FileExistsError(f"Refusing to write into non-empty output path without overwrite: {path}")
        path.mkdir(parents=True, exist_ok=True)
    if config.output_metadata.exists() and not config.overwrite:
        raise FileExistsError(f"Refusing to overwrite metadata without overwrite: {config.output_metadata}")
    config.output_metadata.parent.mkdir(parents=True, exist_ok=True)


def _record_metadata(record: SourceRecord, synthetic_id: str, state: str, transformations: list[dict[str, Any]], config: SynthesisConfig) -> dict[str, Any]:
    return {
        "synthetic_id": synthetic_id,
        "source_image": str(record.image_path),
        "source_id": record.source_id,
        "source_scene_id": record.source_scene_id,
        "source_video": record.source_video,
        "source_frame_index": record.source_frame_index,
        # Kept null for the supplied `fps=1` workflow unless a manifest gives PTS.
        "source_timestamp": record.source_timestamp,
        "timestamp_provenance": "manifest_or_ffprobe" if record.source_timestamp is not None else "unavailable_not_inferred_from_filename",
        "extraction_command": config.extraction_command,
        "confounder_category": record.confounder_category,
        "confounder_score": record.confounder_score,
        "transformation": state,
        "illumination_state": state,
        "severity": [item["residual_illumination"] for item in transformations],
        "selected_light_ids": [item["light_id"] for item in transformations],
        "light_transformations": transformations,
        "random_seed": config.random_seed,
        "generation_version": config.generation_version,
        "timestamp_created": datetime.now(timezone.utc).isoformat(),
        "annotation_action": "copied_unchanged_yolo_detection_labels",
    }


def _report(records: list[SourceRecord], candidates: list[SourceRecord], selected: list[SourceRecord], generated: list[dict[str, Any]], rejected: Counter[str], notes: list[str]) -> dict[str, Any]:
    state_counts = Counter(item["illumination_state"] for item in generated)
    category_counts = Counter(item["confounder_category"] for item in generated)
    return {
        "source_images": len(records), "candidates": len(candidates), "selected_sources": len(selected), "synthetic_images": len(generated),
        "rejected": dict(rejected), "ordinary_candidates": sum(item.confounder_category == "ordinary" for item in candidates),
        "confounder_heavy_candidates": sum(item.confounder_category == "confounder_heavy" for item in candidates),
        "selected_ordinary": sum(item.confounder_category == "ordinary" for item in selected),
        "selected_confounder_heavy": sum(item.confounder_category == "confounder_heavy" for item in selected),
        "generated_ordinary": category_counts["ordinary"], "generated_confounder_heavy": category_counts["confounder_heavy"],
        "actual_confounder_fraction": round(category_counts["confounder_heavy"] / len(generated), 6) if generated else 0.0,
        "illumination_states": dict(state_counts), "synthetic_images_per_source": dict(Counter(item["source_id"] for item in generated)),
        "unique_source_scenes": len({item.source_scene_id for item in selected}), "warnings": notes,
    }


def generate(config: SynthesisConfig, limit: int | None = None) -> dict[str, Any]:
    """Run a pilot/full generation. `limit` is a cap on synthetic images, not sources."""
    records = load_source_records(config)
    split_scenes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_scenes[record.source_scene_id].add(record.split)
    held_out_scenes = {scene for scene, splits in split_scenes.items() if splits & {"valid", "test"}}
    eligible = [record for record in records if record.split == "train" and record.source_scene_id not in held_out_scenes and has_eligible_streetlight(record, config)]
    rejected: Counter[str] = Counter()
    rejected["non_training_or_held_out_scene"] = len(records) - len(eligible)
    candidates, duplicate_counts = filter_candidates(eligible, config)
    rejected.update(duplicate_counts)
    classify_scene_difficulty(candidates, config)
    selected, notes = select_source_frames(candidates, config)
    if config.dry_run:
        report = _report(records, candidates, selected, [], rejected, notes)
        return report
    _prepare_outputs(config)
    metadata: list[dict[str, Any]] = []
    preview_directory = config.output_metadata.parent / "previews"
    for source in selected:
        source_rng = _variant_rng(config.random_seed, source.source_id, -1)
        for variant_index in range(_variant_count(config, source_rng)):
            if limit is not None and len(metadata) >= limit:
                break
            rng = _variant_rng(config.random_seed, source.source_id, variant_index)
            state = _sample_state(rng, config.state_probabilities)
            eligible_lights = _eligible_annotations(source, config)
            if not eligible_lights:
                continue
            count = min(len(eligible_lights), max(1, config.max_lights_per_variant))
            if state != "mixed":
                count = 1
            chosen = rng.sample(eligible_lights, k=rng.randint(1, count))
            with Image.open(source.image_path) as opened:
                original = opened.convert("RGB")
            result = original.copy()
            masks: list[Image.Image] = []
            transformations: list[dict[str, Any]] = []
            for light_id, annotation in chosen:
                light_state = state if state != "mixed" else rng.choice(("dim", "severe_dim", "flicker", "off"))
                residual = _sample_residual(light_state, rng, config)
                mask = create_soft_illumination_mask(original.size, luminous_region(annotation, original.size, config), config.mask_blur_radius_fraction)
                result = apply_off(result, mask, residual) if light_state == "off" else apply_dim(result, mask, residual)
                masks.append(mask)
                transformations.append({"light_id": light_id, "class_id": annotation.class_id, "state": light_state, "residual_illumination": round(residual, 6), "region_source": "dedicated_luminous_class" if annotation.class_id in config.luminous_class_ids else "full_streetlight_top_box_fallback"})
            state_label = state
            severity_label = "mixed" if state == "mixed" else f"{sum(item['residual_illumination'] for item in transformations) / len(transformations):.3f}"
            synthetic_id = f"{_safe_name(source.source_id)}__v{variant_index:02d}__{state_label}__{severity_label}"
            image_path = config.output_images / f"{synthetic_id}.jpg"
            label_path = config.output_annotations / f"{synthetic_id}.txt"
            if (image_path.exists() or label_path.exists()) and not config.overwrite:
                raise FileExistsError(f"Synthetic output already exists: {synthetic_id}; use overwrite only for a deliberate rerun")
            result.save(image_path, quality=95)
            shutil.copyfile(source.annotation_path, label_path)
            entry = _record_metadata(source, synthetic_id, state, transformations, config)
            entry["synthetic_image"] = str(image_path)
            entry["synthetic_annotation"] = str(label_path)
            metadata.append(entry)
            if config.generate_preview and len(metadata) <= config.preview_limit:
                _write_preview(original, combine_masks(masks, original.size), result, preview_directory / f"{synthetic_id}.jpg")
        if limit is not None and len(metadata) >= limit:
            break
    with config.output_metadata.open("w", encoding="utf-8") as handle:
        for entry in metadata:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    report = _report(records, candidates, selected, metadata, rejected, notes)
    report_path = config.output_metadata.parent / "synthetic_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def statistics_from_metadata(metadata_path: Path) -> dict[str, Any]:
    entries = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "synthetic_images": len(entries),
        "illumination_states": dict(Counter(entry["illumination_state"] for entry in entries)),
        "confounder_categories": dict(Counter(entry["confounder_category"] for entry in entries)),
        "unique_source_scenes": len({entry["source_scene_id"] for entry in entries}),
        "synthetic_images_per_source": dict(Counter(entry["source_id"] for entry in entries)),
    }


def probe_frame_timestamps(video_path: Path) -> list[dict[str, Any]]:
    """Optional ffprobe PTS recovery; callers must map returned frames to outputs explicitly."""
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,coded_picture_number", "-of", "json", str(video_path)]
    response = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(response.stdout).get("frames", [])
