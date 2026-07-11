from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from rbccps_annotator.schema import (
    ANNOTATOR_SCHEMA_VERSION,
    IMAGE_EXTENSIONS,
    LAMP_BOX_CLASSES,
    AnnotationItem,
    TrackRecord,
    WorkspaceManifest,
    normalize_lamp_class,
    sanitize_key,
)
from rbccps_annotator.workspace import ensure_dir, write_csv, write_json

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}
SUPPORTED_RAW_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass(frozen=True)
class BundlePrepareResult:
    workspace: Path
    sampled_frames: Path
    exports: Path
    candidate_count: int
    sampled_count: int
    tutorial_count: int
    detector_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "sampled_frames": str(self.sampled_frames),
            "exports": str(self.exports),
            "candidate_count": self.candidate_count,
            "sampled_count": self.sampled_count,
            "tutorial_count": self.tutorial_count,
            "detector_used": self.detector_used,
        }


@dataclass(frozen=True)
class CandidateFrame:
    path: Path
    source_path: Path
    source_type: str
    clip_id: str
    source_frame_index: int
    source_timestamp_s: float | None


@dataclass(frozen=True)
class FrameFeature:
    candidate: CandidateFrame
    luma_mean: float
    luma_std: float
    bright_fraction: float
    edge_mean: float
    ahash: int
    width: int
    height: int

    @property
    def diversity_vector(self) -> tuple[float, float, float, float]:
        return (
            self.luma_mean / 255.0,
            self.luma_std / 128.0,
            min(1.0, self.bright_fraction * 20.0),
            self.edge_mean / 64.0,
        )


def prepare_bundle_workspace(
    bundle_root: Path,
    input_raw: Path | None = None,
    workspace: Path | None = None,
    batch_id: str | None = None,
    sample_budget: int | None = None,
    fps: float = 1.0,
    detector_weights: Path | None = None,
    tutorial_examples: Path | None = None,
    force: bool = False,
) -> BundlePrepareResult:
    bundle_root = bundle_root.resolve()
    input_raw = (input_raw or bundle_root / "input_raw").resolve()
    batch_id = sanitize_key(batch_id or time.strftime("batch_%Y%m%d_%H%M%S"))
    workspace = (workspace or bundle_root / "workspaces" / batch_id).resolve()
    sampled_frames = workspace / "sampled_frames"
    candidates_dir = workspace / "_candidate_frames"
    logs_dir = ensure_dir(bundle_root / "logs")
    exports_dir = ensure_dir(bundle_root / "exports" / batch_id)

    if workspace.exists() and force:
        shutil.rmtree(workspace)
    if workspace.exists() and (workspace / "manifest.json").exists() and not force:
        tutorial_count = validate_tutorial_examples(tutorial_examples or bundle_root / "tutorial_examples", workspace)
        return BundlePrepareResult(workspace, sampled_frames, exports_dir, 0, _count_images(sampled_frames), tutorial_count, False)

    ensure_dir(workspace)
    ensure_dir(sampled_frames)
    ensure_dir(candidates_dir)

    raw_files = scan_raw_media(input_raw)
    if not raw_files:
        raise ValueError(f"No supported raw images/videos found in {input_raw}")

    candidates = materialize_candidate_frames(raw_files, candidates_dir, fps=fps, log_path=logs_dir / f"{batch_id}_ffmpeg.log")
    if not candidates:
        raise ValueError("No candidate frames could be created from the raw inputs.")

    budget = sample_budget or auto_sample_budget(len(candidates))
    selected = smart_sample(candidates, budget)
    copied = copy_sampled_frames(selected, sampled_frames)
    tracks_by_key = prelabel_with_detector(copied, detector_weights or bundle_root / "models" / "detector" / "best.pt")
    write_workspace_from_sampled_frames(copied, workspace, batch_id, tracks_by_key)
    write_sampling_manifest(workspace / "sampling_manifest.csv", copied)
    tutorial_count = validate_tutorial_examples(tutorial_examples or bundle_root / "tutorial_examples", workspace)
    write_json(
        workspace / "bundle_state.json",
        {
            "batch_id": batch_id,
            "input_raw": str(input_raw),
            "exports": str(exports_dir),
            "tutorial_completed": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "candidate_count": len(candidates),
            "sampled_count": len(copied),
            "detector_used": bool(tracks_by_key),
        },
    )
    return BundlePrepareResult(workspace, sampled_frames, exports_dir, len(candidates), len(copied), tutorial_count, bool(tracks_by_key))


def scan_raw_media(input_raw: Path) -> list[Path]:
    if not input_raw.exists():
        return []
    return sorted(path for path in input_raw.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_RAW_EXTENSIONS)


def materialize_candidate_frames(raw_files: list[Path], candidates_dir: Path, fps: float, log_path: Path | None = None) -> list[CandidateFrame]:
    rows: list[CandidateFrame] = []
    for source in raw_files:
        clip_id = sanitize_key(source.stem)
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            target = candidates_dir / clip_id / f"{clip_id}_image_000001{source.suffix.lower()}"
            ensure_dir(target.parent)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            rows.append(CandidateFrame(target, source, "image", clip_id, 1, None))
        elif source.suffix.lower() in VIDEO_EXTENSIONS:
            rows.extend(extract_video_candidates(source, candidates_dir / clip_id, fps=fps, log_path=log_path))
    return rows


def extract_video_candidates(video: Path, output_dir: Path, fps: float = 1.0, log_path: Path | None = None) -> list[CandidateFrame]:
    ensure_dir(output_dir)
    output_pattern = output_dir / f"{sanitize_key(video.stem)}_%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required for raw video ingestion. Install ffmpeg or place only images in input_raw.") from error
    if log_path:
        ensure_dir(log_path.parent)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("COMMAND " + " ".join(command) + "\n")
            if completed.stdout:
                handle.write(completed.stdout + "\n")
            if completed.stderr:
                handle.write(completed.stderr + "\n")
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video}: {completed.stderr.strip()}")
    frames = sorted(output_dir.glob("*.jpg"))
    return [
        CandidateFrame(path=frame, source_path=video, source_type="video", clip_id=sanitize_key(video.stem), source_frame_index=index, source_timestamp_s=(index - 1) / max(fps, 0.001))
        for index, frame in enumerate(frames, start=1)
    ]


def auto_sample_budget(candidate_count: int) -> int:
    if candidate_count <= 150:
        return candidate_count
    return max(300, min(600, int(math.sqrt(candidate_count) * 35)))


def smart_sample(candidates: list[CandidateFrame], budget: int) -> list[CandidateFrame]:
    features = [feature_frame(candidate) for candidate in candidates]
    features = [feature for feature in features if feature is not None]
    if len(features) <= budget:
        return [feature.candidate for feature in features]

    by_clip: dict[str, list[FrameFeature]] = {}
    for feature in features:
        by_clip.setdefault(feature.candidate.clip_id, []).append(feature)
    per_clip_min = max(1, budget // max(1, len(by_clip)))
    selected: list[FrameFeature] = []
    seen_hashes: list[int] = []

    for clip_features in by_clip.values():
        selected.extend(_select_diverse(clip_features, min(per_clip_min, len(clip_features)), seen_hashes))
    if len(selected) < budget:
        remaining = [feature for feature in features if feature not in selected]
        selected.extend(_select_diverse(remaining, budget - len(selected), seen_hashes))
    selected = sorted(selected[:budget], key=lambda item: (item.candidate.clip_id, item.candidate.source_frame_index))
    return [feature.candidate for feature in selected]


def feature_frame(candidate: CandidateFrame) -> FrameFeature | None:
    try:
        with Image.open(candidate.path) as image:
            gray = image.convert("L")
            small = gray.resize((8, 8))
            stat = ImageStat.Stat(gray)
            edge = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edge)
            pixels = list(small.tobytes())
            mean_small = sum(pixels) / len(pixels)
            ahash = 0
            for index, value in enumerate(pixels):
                if value >= mean_small:
                    ahash |= 1 << index
            hist = gray.histogram()
            total = max(1, gray.width * gray.height)
            bright = sum(hist[220:]) / total
            return FrameFeature(
                candidate=candidate,
                luma_mean=float(stat.mean[0]),
                luma_std=float(stat.stddev[0]),
                bright_fraction=float(bright),
                edge_mean=float(edge_stat.mean[0]),
                ahash=ahash,
                width=gray.width,
                height=gray.height,
            )
    except Exception:
        return None


def _select_diverse(features: list[FrameFeature], count: int, seen_hashes: list[int]) -> list[FrameFeature]:
    if count <= 0 or not features:
        return []
    ranked = sorted(features, key=lambda item: (item.bright_fraction, item.luma_std, item.edge_mean), reverse=True)
    selected: list[FrameFeature] = []
    for feature in ranked:
        if len(selected) >= count:
            break
        if any(_hamming(feature.ahash, existing) <= 5 for existing in seen_hashes):
            continue
        selected.append(feature)
        seen_hashes.append(feature.ahash)
    if len(selected) < count:
        for feature in ranked:
            if feature not in selected:
                selected.append(feature)
                if len(selected) >= count:
                    break
    return selected


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def copy_sampled_frames(selected: list[CandidateFrame], sampled_frames: Path) -> list[CandidateFrame]:
    ensure_dir(sampled_frames)
    copied: list[CandidateFrame] = []
    for index, candidate in enumerate(selected, start=1):
        suffix = candidate.path.suffix.lower() or ".jpg"
        target = sampled_frames / f"{index:06d}_{sanitize_key(candidate.clip_id)}{suffix}"
        shutil.copy2(candidate.path, target)
        copied.append(
            CandidateFrame(
                path=target,
                source_path=candidate.source_path,
                source_type=candidate.source_type,
                clip_id=candidate.clip_id,
                source_frame_index=candidate.source_frame_index,
                source_timestamp_s=candidate.source_timestamp_s,
            )
        )
    return copied


def prelabel_with_detector(frames: list[CandidateFrame], detector_weights: Path) -> dict[str, list[TrackRecord]]:
    if not detector_weights.exists():
        return {}
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return {}
    model = YOLO(str(detector_weights))
    tracks_by_key: dict[str, list[TrackRecord]] = {}
    for index, frame in enumerate(frames, start=1):
        key = sampled_item_key("portable", frame, index)
        try:
            result = model(str(frame.path), verbose=False)[0]
        except Exception:
            continue
        records: list[TrackRecord] = []
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box_index, box in enumerate(boxes, start=1):
            try:
                class_id = int(box.cls.item())
                class_name = normalize_lamp_class(names.get(class_id, "streetlight_lamp_head"))
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                score = float(box.conf.item())
            except Exception:
                continue
            if class_name not in LAMP_BOX_CLASSES:
                class_name = "streetlight_lamp_head"
            records.append(
                TrackRecord(
                    track_id=f"det_{index:06d}_{box_index:02d}",
                    class_name=class_name,
                    bbox_xyxy=xyxy,
                    detector_score=score,
                    track_confidence=score,
                    track_age=1,
                    lost_count=0,
                    source_model=str(detector_weights),
                )
            )
        if records:
            tracks_by_key[key] = records
    return tracks_by_key


def sampled_item_key(dataset_id: str, frame: CandidateFrame, index: int) -> str:
    return sanitize_key(f"{dataset_id}_{frame.clip_id}_{index:06d}_{frame.path.stem}")


def write_workspace_from_sampled_frames(frames: list[CandidateFrame], workspace: Path, batch_id: str, tracks_by_key: dict[str, list[TrackRecord]] | None = None) -> Path:
    tracks_by_key = tracks_by_key or {}
    items: list[AnnotationItem] = []
    split_by_clip = assign_clip_splits(sorted({frame.clip_id for frame in frames}))
    for index, frame in enumerate(frames, start=1):
        with Image.open(frame.path) as image:
            width, height = image.size
        key = sampled_item_key("portable", frame, index)
        items.append(
            AnnotationItem(
                key=key,
                image_id=sanitize_key(frame.path.stem),
                image_path=str(frame.path.resolve()),
                width=width,
                height=height,
                dataset_id="portable_bundle",
                route_id="",
                clip_id=frame.clip_id,
                frame_id=str(frame.source_frame_index),
                timestamp_ns=int((frame.source_timestamp_s or index) * 1_000_000_000),
                split=split_by_clip.get(frame.clip_id, "train"),
                source_pool=frame.source_type,
                tracks=tracks_by_key.get(key, []),
                metadata={
                    "source_path": str(frame.source_path.resolve()),
                    "source_type": frame.source_type,
                    "source_timestamp_s": frame.source_timestamp_s,
                    "metadata_quality": "video_image_only",
                    "missing_metadata_flags": ["no_camera2_metadata", "no_gps_imu_sidecar", "no_lux_reference"],
                },
            )
        )
    manifest = WorkspaceManifest(
        schema_version=ANNOTATOR_SCHEMA_VERSION,
        workspace_id=sanitize_key(workspace.name),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_type="portable_raw_media",
        notes=f"Portable annotator bundle batch {batch_id}",
        items=items,
    )
    write_json(workspace / "manifest.json", manifest.to_dict())
    _write_frames_csv_with_metadata(workspace / "frames.csv", items)
    return workspace


def assign_clip_splits(clip_ids: list[str]) -> dict[str, str]:
    if not clip_ids:
        return {}
    splits: dict[str, str] = {}
    for index, clip_id in enumerate(clip_ids):
        bucket = index % 10
        if bucket < 7:
            split = "train"
        elif bucket < 9:
            split = "valid"
        else:
            split = "test"
        splits[clip_id] = split
    return splits


def write_sampling_manifest(path: Path, frames: list[CandidateFrame]) -> None:
    rows = [
        {
            "sampled_path": str(frame.path),
            "source_path": str(frame.source_path),
            "source_type": frame.source_type,
            "clip_id": frame.clip_id,
            "source_frame_index": frame.source_frame_index,
            "source_timestamp_s": frame.source_timestamp_s if frame.source_timestamp_s is not None else "",
        }
        for frame in frames
    ]
    write_csv(path, rows, ["sampled_path", "source_path", "source_type", "clip_id", "source_frame_index", "source_timestamp_s"])


def validate_tutorial_examples(tutorial_dir: Path, workspace: Path | None = None) -> int:
    if not tutorial_dir.exists():
        if workspace:
            write_json(workspace / "tutorial_manifest.json", {"examples": [], "warnings": ["tutorial_examples directory not found"]})
        return 0
    examples = []
    warnings = []
    for json_path in sorted(tutorial_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except Exception as error:
            warnings.append(f"{json_path.name}: invalid JSON: {error}")
            continue
        image_name = payload.get("image") or payload.get("image_path")
        if not image_name:
            warnings.append(f"{json_path.name}: missing image field")
            continue
        image_path = (tutorial_dir / str(image_name)).resolve()
        if not image_path.exists():
            warnings.append(f"{json_path.name}: missing image {image_name}")
            continue
        review = payload.get("review") or payload
        if not isinstance(review, dict) or "boxes" not in review:
            warnings.append(f"{json_path.name}: missing review.boxes")
            continue
        for box in review.get("boxes", []):
            box["class_name"] = normalize_lamp_class(box.get("class_name"))
        examples.append(
            {
                "id": sanitize_key(payload.get("id") or json_path.stem),
                "title": payload.get("title") or json_path.stem.replace("_", " ").title(),
                "lesson": payload.get("lesson") or "",
                "image_path": str(image_path),
                "review_path": str(json_path.resolve()),
                "review": review,
            }
        )
    if workspace:
        write_json(workspace / "tutorial_manifest.json", {"examples": examples, "warnings": warnings})
    if warnings and not examples:
        raise ValueError("Tutorial examples are present but invalid: " + "; ".join(warnings))
    return len(examples)


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def _write_frames_csv_with_metadata(path: Path, items: list[AnnotationItem]) -> None:
    rows = []
    for item in items:
        rows.append(
            {
                "key": item.key,
                "image_id": item.image_id,
                "image_path": item.image_path,
                "dataset_id": item.dataset_id,
                "route_id": item.route_id,
                "clip_id": item.clip_id,
                "frame_id": item.frame_id,
                "timestamp_ns": item.timestamp_ns or "",
                "width": item.width,
                "height": item.height,
                "split": item.split,
                "source_pool": item.source_pool,
                "track_count": len(item.tracks),
                "source_path": item.metadata.get("source_path", ""),
                "source_timestamp_s": item.metadata.get("source_timestamp_s", ""),
                "metadata_quality": item.metadata.get("metadata_quality", ""),
                "missing_metadata_flags": ";".join(item.metadata.get("missing_metadata_flags", [])),
            }
        )
    fieldnames = list(rows[0].keys()) if rows else ["key"]
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
