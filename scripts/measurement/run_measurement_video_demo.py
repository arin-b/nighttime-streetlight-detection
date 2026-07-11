from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from rbccps_measurement.contracts.input_schema import ClipManifest
from rbccps_measurement.ingest.validation import validate_clip_manifest
from rbccps_measurement.models.registry import get_asset
from rbccps_measurement.pipeline import run_clip_to_directory


DEFAULT_FRAMES_DIR = Path("datasets/extracted_frames/mobile_night_videos/2025-05-29/20250529_2050207")
IMPLEMENTATION = "untrained_deterministic_video_demo_v1"


@dataclass(frozen=True)
class FrameDetection:
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int = 0


@dataclass
class LinkedTrack:
    track_id: str
    last_bbox: tuple[float, float, float, float]
    last_frame_index: int
    detections: list[FrameDetection] = field(default_factory=list)


def repo_root() -> Path:
    return ROOT


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def link_detections(
    detections_by_frame: list[list[FrameDetection]],
    iou_threshold: float = 0.35,
    max_frame_gap: int = 3,
) -> list[LinkedTrack]:
    active: list[LinkedTrack] = []
    finished: list[LinkedTrack] = []
    next_id = 1
    for frame_index, detections in enumerate(detections_by_frame, start=1):
        still_active: list[LinkedTrack] = []
        for track in active:
            if frame_index - track.last_frame_index <= max_frame_gap:
                still_active.append(track)
            else:
                finished.append(track)
        active = still_active
        used_tracks: set[str] = set()
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            best_track: LinkedTrack | None = None
            best_iou = 0.0
            for track in active:
                if track.track_id in used_tracks:
                    continue
                overlap = bbox_iou(track.last_bbox, detection.bbox_xyxy)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track = track
            if best_track is not None and best_iou >= iou_threshold:
                best_track.detections.append(detection)
                best_track.last_bbox = detection.bbox_xyxy
                best_track.last_frame_index = frame_index
                used_tracks.add(best_track.track_id)
            else:
                track = LinkedTrack(
                    track_id=f"demo_lamp_{next_id:04d}",
                    last_bbox=detection.bbox_xyxy,
                    last_frame_index=frame_index,
                    detections=[detection],
                )
                next_id += 1
                active.append(track)
                used_tracks.add(track.track_id)
    return [*finished, *active]


def run_yolo_on_frames(
    frame_paths: list[Path],
    model_path: Path,
    conf: float,
    max_det: int,
    batch_size: int = 4,
) -> list[list[FrameDetection]]:
    yolo = YOLO(str(model_path))
    detections_by_frame: list[list[FrameDetection]] = []
    chunk_size = max(1, int(batch_size))
    for start in range(0, len(frame_paths), chunk_size):
        chunk = frame_paths[start : start + chunk_size]
        results = yolo.predict(
            [str(path) for path in chunk],
            conf=conf,
            iou=0.55,
            max_det=max_det,
            verbose=False,
            batch=chunk_size,
        )
        for offset, (frame_path, result) in enumerate(zip(chunk, results), start=1):
            frame_index = start + offset
            image = Image.open(frame_path).convert("RGB")
            width, height = image.size
            detections: list[FrameDetection] = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    x1 = max(0.0, min(width - 1.0, x1))
                    y1 = max(0.0, min(height - 1.0, y1))
                    x2 = max(x1 + 1.0, min(float(width), x2))
                    y2 = max(y1 + 1.0, min(float(height), y2))
                    detections.append(
                        FrameDetection(
                            frame_index=frame_index,
                            bbox_xyxy=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                            score=round(float(box.conf[0].item()), 4),
                            class_id=int(box.cls[0].item()) if box.cls is not None else 0,
                        )
                    )
            detections_by_frame.append(detections)
    return detections_by_frame


def build_clip_manifest_payload(
    frame_paths: list[Path],
    linked_tracks: list[LinkedTrack],
    out: Path,
    fps: float,
    clip_id: str = "untrained_measurement_video_demo",
) -> dict[str, Any]:
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    copied_frames: list[Path] = []
    for index, source in enumerate(frame_paths, start=1):
        target = frames_dir / f"{index:06d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied_frames.append(target)
    first = Image.open(copied_frames[0]).convert("RGB")
    width, height = first.size
    timestamp0 = 1_700_000_000_000_000_000
    step_ns = int(1_000_000_000 / max(0.1, fps))
    frame_records = [
        {
            "frame_id": index,
            "timestamp_ns": timestamp0 + (index - 1) * step_ns,
            "image_uri": "frames/" + copied_frames[index - 1].name,
            "image_format": copied_frames[index - 1].suffix.lower().lstrip("."),
            "width": width,
            "height": height,
            "camera": {
                "exposure_time_s": None,
                "sensor_sensitivity_iso": None,
                "ae_mode": "auto",
                "hdr_mode": "unknown",
                "night_mode": True,
                "metadata_quality": "video_image_only",
            },
            "pose": {
                "latitude": None,
                "longitude": None,
                "gps_accuracy_m": None,
                "heading_deg": None,
                "imu_quality": "missing",
            },
        }
        for index in range(1, len(copied_frames) + 1)
    ]
    track_records: list[dict[str, Any]] = []
    for track in linked_tracks:
        for detection in track.detections:
            track_records.append(
                {
                    "frame_id": detection.frame_index,
                    "timestamp_ns": timestamp0 + (detection.frame_index - 1) * step_ns,
                    "track_id": track.track_id,
                    "class_name": "streetlight_lamp_head",
                    "bbox_xyxy": list(detection.bbox_xyxy),
                    "bbox_format": "pixel_xyxy_original_frame",
                    "detector_score": detection.score,
                    "track_confidence": detection.score,
                    "track_age": len(track.detections),
                    "lost_count": 0,
                    "source_model": "streetlight_detector_v3:hpc_pull",
                    "optional_cue_scores": {"yolo_class_id": float(detection.class_id), "demo_conf_threshold": 0.0},
                }
            )
    track_records.sort(key=lambda row: (row["frame_id"], row["track_id"]))
    return {
        "clip_id": clip_id,
        "device_id": "untrained_demo_extracted_frames",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "video_uri": None,
        "frames": frame_records,
        "tracks": track_records,
        "optional_calibration": {
            "photometric": {"field_lux_calibration_id": None},
            "map_priors": {"route_group": "untrained_demo"},
        },
    }


def render_processed_frames(
    out: Path,
    manifest: dict[str, Any],
    reports_by_track: dict[str, dict[str, Any]],
    fps: float,
) -> None:
    processed_dir = out / "processed_frames"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tracks_by_frame: dict[int, list[dict[str, Any]]] = {}
    for track in manifest["tracks"]:
        tracks_by_frame.setdefault(int(track["frame_id"]), []).append(track)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    output_frames: list[Path] = []
    for frame in manifest["frames"]:
        frame_id = int(frame["frame_id"])
        image_path = out / frame["image_uri"]
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 58), fill=(0, 0, 0))
        draw.text((12, 8), f"RBCCPS untrained deterministic demo | frame {frame_id} | fps {fps:g}", fill=(255, 255, 255), font=font)
        draw.text((12, 34), "YOLO detections + measurement-block proxy outputs; not trained/calibrated performance", fill=(255, 220, 120), font=small)
        for track in tracks_by_frame.get(frame_id, []):
            x1, y1, x2, y2 = [float(v) for v in track["bbox_xyxy"]]
            report = reports_by_track.get(track["track_id"], {})
            category = report.get("metrics", {}).get("overall_category", "no_report")
            score = report.get("metrics", {}).get("overall_useful_illumination_score")
            status = report.get("status", {}).get("label", "")
            action = report.get("confidence", {}).get("action", "")
            color = _category_color(category)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            label = f"{track['track_id']} det={track['detector_score']:.2f} {category}"
            if score is not None:
                label += f" s={score:.2f}"
            if status:
                label += f" {status}"
            if action:
                label += f" {action}"
            text_y = max(60, y1 - 18)
            draw.rectangle((x1, text_y - 2, min(image.width - 1, x1 + 360), text_y + 16), fill=(0, 0, 0))
            draw.text((x1 + 3, text_y), label, fill=color, font=small)
        target = processed_dir / f"processed_{frame_id:06d}.png"
        image.save(target)
        output_frames.append(target)
    _write_mp4(output_frames, out / "processed_demo.mp4", fps=fps)
    _write_contact_sheet(output_frames, out / "contact_sheet.png")


def run_demo(
    frames_dir: Path,
    out: Path,
    fps: float,
    max_frames: int,
    conf: float,
    iou_link_threshold: float,
    max_det: int,
    batch_size: int = 4,
) -> dict[str, Any]:
    repo = repo_root()
    frames_dir = frames_dir if frames_dir.is_absolute() else repo / frames_dir
    out = out if out.is_absolute() else repo / out
    out.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:max_frames]
    if not frame_paths:
        raise ValueError(f"no demo frames found under {frames_dir}")
    model_path = repo / get_asset("streetlight_detector_v3").local_path
    detections_by_frame = run_yolo_on_frames(frame_paths, model_path, conf=conf, max_det=max_det, batch_size=batch_size)
    linked_tracks = link_detections(detections_by_frame, iou_threshold=iou_link_threshold)
    manifest = build_clip_manifest_payload(frame_paths, linked_tracks, out, fps=fps)
    manifest_path = out / "clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    measurement_dir = out / "measurement"
    reports = []
    if manifest["tracks"]:
        validate_clip_manifest(ClipManifest.from_dict(manifest))
        reports = run_clip_to_directory(manifest_path, measurement_dir, measurement_run_id="untrained_deterministic_demo")
    reports_by_track = {report.lamp_track_id: report.to_dict() for report in reports}
    render_processed_frames(out, manifest, reports_by_track, fps=fps)
    summary = {
        "implementation": IMPLEMENTATION,
        "frames_dir": str(frames_dir),
        "frames": len(frame_paths),
        "fps": fps,
        "duration_seconds": round(len(frame_paths) / fps, 3),
        "detector_model": str(model_path),
        "conf_threshold": conf,
        "iou_link_threshold": iou_link_threshold,
        "max_det": max_det,
        "batch_size": batch_size,
        "detections": sum(len(items) for items in detections_by_frame),
        "linked_tracks": len(linked_tracks),
        "measurement_reports": len(reports),
        "untrained_notice": "Qualitative deterministic demo only; these outputs are not trained or calibrated performance claims.",
        "artifacts": {
            "manifest": str(manifest_path),
            "measurement_reports": str(measurement_dir / "reports.json") if reports else None,
            "processed_video": str(out / "processed_demo.mp4"),
            "contact_sheet": str(out / "contact_sheet.png"),
            "processed_frames": str(out / "processed_frames"),
        },
    }
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _category_color(category: str) -> tuple[int, int, int]:
    return {
        "adequate": (52, 211, 153),
        "marginal": (250, 204, 21),
        "poor": (251, 146, 60),
        "unknown": (248, 113, 113),
    }.get(str(category), (147, 197, 253))


def _write_mp4(frames: list[Path], path: Path, fps: float) -> None:
    if not frames:
        return
    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            image = cv2.imread(str(frame))
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()


def _write_contact_sheet(frames: list[Path], path: Path, columns: int = 4, rows: int = 4) -> None:
    selected = frames[:: max(1, len(frames) // (columns * rows))][: columns * rows]
    thumbs: list[Image.Image] = []
    for frame in selected:
        image = Image.open(frame).convert("RGB")
        image.thumbnail((360, 220), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 220), (20, 20, 20))
        canvas.paste(image, ((360 - image.width) // 2, (220 - image.height) // 2))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (columns * 360, rows * 220), (35, 35, 35))
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 360
        y = (index // columns) * 220
        sheet.paste(thumb, (x, y))
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO + RBCCPS measurement pipeline on an extracted frame sequence and render an untrained demo video.")
    parser.add_argument("--frames-dir", default=str(DEFAULT_FRAMES_DIR), help="Directory of extracted frames.")
    parser.add_argument("--out", required=True, help="Output directory for demo artifacts.")
    parser.add_argument("--fps", type=float, default=3.0, help="Output video FPS.")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to process.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold.")
    parser.add_argument("--iou-link-threshold", type=float, default=0.30, help="IoU threshold for simple demo track linking.")
    parser.add_argument("--max-det", type=int, default=8, help="Max detections per frame.")
    parser.add_argument("--batch-size", type=int, default=4, help="YOLO batch size for frame inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_demo(
        frames_dir=Path(args.frames_dir),
        out=Path(args.out),
        fps=args.fps,
        max_frames=args.max_frames,
        conf=args.conf,
        iou_link_threshold=args.iou_link_threshold,
        max_det=args.max_det,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
