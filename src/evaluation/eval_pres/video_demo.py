from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .artifact_utils import repo_root, write_csv, write_json
from .cli import evaluate_to_directory
from .detection_tracking_artifacts import bbox_iou, build_detection_tracking_artifacts
from .measurement_artifacts import build_measurement_artifacts
from .run_metadata import write_run_sidecars
from .video_utils import extract_frames, write_contact_sheet, write_mp4, write_representative_frames


DEFAULT_VIDEO = Path("eval_pres/sample_videos/busy_street_20260220_200501_629_1min.mp4")
DEFAULT_MODEL = Path("models/measurement/pretrained/streetlight_detector_v3/hpc_pull/best.pt")


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


def run_video_demo(
    video: str | Path,
    out: str | Path,
    fps_sample: float = 3.0,
    conf: float = 0.05,
    max_frames: int | None = None,
    max_det: int = 12,
    iou_link_threshold: float = 0.30,
    model_path: str | Path | None = None,
    ground_truth: str | Path | None = None,
    route_distance_km: float | None = None,
    skip_detector: bool = False,
    measurement_max_tracks: int | None = 30,
    preset: str | None = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    root = repo_root()
    video_path = Path(video)
    if not video_path.is_absolute():
        video_path = root / video_path
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    frame_dir = out_path / "frames"
    frame_paths = extract_frames(video_path, frame_dir, fps_sample=fps_sample, max_frames=max_frames)
    if not frame_paths:
        raise ValueError(f"no frames extracted from {video_path}")

    resolved_model = Path(model_path) if model_path else root / DEFAULT_MODEL
    if not resolved_model.is_absolute():
        resolved_model = root / resolved_model
    if skip_detector:
        detections_by_frame: list[list[FrameDetection]] = [[] for _ in frame_paths]
    else:
        detections_by_frame = run_detector(frame_paths, resolved_model, conf=conf, max_det=max_det)
    linked_tracks_all = link_detections(detections_by_frame, iou_threshold=iou_link_threshold)
    linked_tracks = select_measurement_tracks(linked_tracks_all, measurement_max_tracks)
    manifest = build_clip_manifest(frame_paths, linked_tracks, fps_sample=fps_sample, clip_id=video_path.stem)
    manifest_path = out_path / "clip_manifest.json"
    write_json(manifest_path, manifest)
    write_detection_tables(out_path, detections_by_frame, linked_tracks_all)

    measurement_dir = out_path / "measurement"
    reports = run_measurement(manifest_path, measurement_dir)
    report_dicts = [report.to_dict() for report in reports]

    processed_frames = render_processed_frames(
        manifest=manifest,
        reports=report_dicts,
        frame_root=out_path,
        out_dir=out_path / "processed_frames",
    )
    processed_video = write_mp4(processed_frames, out_path / "processed_video.mp4", fps_sample) if processed_frames else None
    contact_sheet = write_contact_sheet(processed_frames, out_path / "contact_sheet.jpg") if processed_frames else None
    representative_dir = write_representative_frames(processed_frames, out_path / "representative_frames")

    detection_artifacts = build_detection_tracking_artifacts(
        manifest_path=manifest_path,
        out_dir=out_path / "detection_tracking_artifacts",
        frame_root=out_path,
        ground_truth_path=ground_truth,
        render_overlays=True,
        video_fps=fps_sample,
    )
    measurement_artifacts = build_measurement_artifacts(
        reports_path=measurement_dir / "reports.json",
        out_dir=out_path / "measurement_artifacts",
        manifest_path=manifest_path,
        frame_root=out_path,
        measurement_dir=measurement_dir,
    )
    elapsed = time.perf_counter() - start_time
    evaluation = evaluate_to_directory(
        manifest=manifest_path,
        reports=measurement_dir / "reports.json",
        out=out_path / "evaluation",
        ground_truth=ground_truth,
        route_distance_km=route_distance_km,
        latency_seconds=elapsed,
        model_paths=[resolved_model],
        run_name="eval_pres_video_demo_raw_pretrained",
    )
    summary = {
        "implementation": "eval_pres_video_demo_v1",
        "preset": preset,
        "video": str(video_path),
        "out": str(out_path),
        "fps_sample": fps_sample,
        "confidence_threshold": conf,
        "model_path": str(resolved_model),
        "frames": len(frame_paths),
        "detections": sum(len(items) for items in detections_by_frame),
        "linked_tracks": len(linked_tracks_all),
        "measurement_tracks_used": len(linked_tracks),
        "measurement_track_cap": measurement_max_tracks,
        "measurement_reports": len(report_dicts),
        "elapsed_seconds": round(elapsed, 3),
        "artifacts": {
            "clip_manifest": str(manifest_path),
            "detections_csv": str(out_path / "detections.csv"),
            "detections_json": str(out_path / "detections.json"),
            "tracks_csv": str(out_path / "tracks.csv"),
            "tracks_json": str(out_path / "tracks.json"),
            "measurement": str(measurement_dir),
            "processed_frames": str(out_path / "processed_frames"),
            "representative_frames": str(representative_dir),
            "processed_video": str(processed_video) if processed_video else None,
            "contact_sheet": str(contact_sheet) if contact_sheet else None,
            "detection_tracking_artifacts": detection_artifacts,
            "measurement_artifacts": measurement_artifacts,
            "evaluation_summary": str(out_path / "evaluation" / "evaluation_summary.json"),
        },
        "evaluation_computed_metrics": sum(metric["status"] == "computed" for metric in evaluation["metrics"]),
    }
    write_json(out_path / "demo_summary.json", summary)
    sidecars = write_run_sidecars(
        out_path,
        run_type="demo-video",
        parameters={
            "video": str(video_path),
            "fps_sample": fps_sample,
            "conf": conf,
            "max_frames": max_frames,
            "max_det": max_det,
            "iou_link_threshold": iou_link_threshold,
            "model_path": str(resolved_model),
            "ground_truth": str(ground_truth) if ground_truth else None,
            "route_distance_km": route_distance_km,
            "skip_detector": skip_detector,
            "measurement_max_tracks": measurement_max_tracks,
            "preset": preset,
        },
        summary=summary,
    )
    summary["artifacts"]["run_sidecars"] = sidecars
    write_json(out_path / "demo_summary.json", summary)
    return summary


def run_detector(frame_paths: list[Path], model_path: Path, conf: float, max_det: int) -> list[list[FrameDetection]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to run detector-backed eval_pres video demos") from exc
    model = YOLO(str(model_path))
    detections_by_frame: list[list[FrameDetection]] = []
    for index, frame_path in enumerate(frame_paths, start=1):
        image = Image.open(frame_path).convert("RGB")
        width, height = image.size
        result = model.predict(str(frame_path), conf=conf, iou=0.55, max_det=max_det, verbose=False)[0]
        detections: list[FrameDetection] = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    FrameDetection(
                        frame_index=index,
                        bbox_xyxy=(
                            round(max(0.0, min(width - 1.0, x1)), 2),
                            round(max(0.0, min(height - 1.0, y1)), 2),
                            round(max(1.0, min(float(width), x2)), 2),
                            round(max(1.0, min(float(height), y2)), 2),
                        ),
                        score=round(float(box.conf[0].item()), 4),
                        class_id=int(box.cls[0].item()) if box.cls is not None else 0,
                    )
                )
        detections_by_frame.append(detections)
    return detections_by_frame


def link_detections(detections_by_frame: list[list[FrameDetection]], iou_threshold: float = 0.30, max_frame_gap: int = 3) -> list[LinkedTrack]:
    active: list[LinkedTrack] = []
    finished: list[LinkedTrack] = []
    next_id = 1
    for frame_index, detections in enumerate(detections_by_frame, start=1):
        still_active = []
        for track in active:
            if frame_index - track.last_frame_index <= max_frame_gap:
                still_active.append(track)
            else:
                finished.append(track)
        active = still_active
        used: set[str] = set()
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            best_track = None
            best_iou = 0.0
            for track in active:
                if track.track_id in used:
                    continue
                overlap = bbox_iou(track.last_bbox, detection.bbox_xyxy)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track = track
            if best_track is not None and best_iou >= iou_threshold:
                best_track.detections.append(detection)
                best_track.last_bbox = detection.bbox_xyxy
                best_track.last_frame_index = frame_index
                used.add(best_track.track_id)
            else:
                track = LinkedTrack(
                    track_id=f"eval_lamp_{next_id:04d}",
                    last_bbox=detection.bbox_xyxy,
                    last_frame_index=frame_index,
                    detections=[detection],
                )
                next_id += 1
                active.append(track)
                used.add(track.track_id)
    return [*finished, *active]


def select_measurement_tracks(tracks: list[LinkedTrack], max_tracks: int | None) -> list[LinkedTrack]:
    if max_tracks is None or max_tracks <= 0 or len(tracks) <= max_tracks:
        return tracks

    def rank(track: LinkedTrack) -> tuple[float, int]:
        mean_score = float(np.mean([detection.score for detection in track.detections])) if track.detections else 0.0
        return (len(track.detections) * mean_score, len(track.detections))

    return sorted(tracks, key=rank, reverse=True)[:max_tracks]


def build_clip_manifest(frame_paths: list[Path], linked_tracks: list[LinkedTrack], fps_sample: float, clip_id: str) -> dict[str, Any]:
    first = Image.open(frame_paths[0]).convert("RGB")
    width, height = first.size
    timestamp0 = 1_700_000_000_000_000_000
    step_ns = int(1_000_000_000 / max(0.1, fps_sample))
    frames = [
        {
            "frame_id": index,
            "timestamp_ns": timestamp0 + (index - 1) * step_ns,
            "image_uri": "frames/" + frame_paths[index - 1].name,
            "image_format": frame_paths[index - 1].suffix.lower().lstrip("."),
            "width": width,
            "height": height,
            "camera": {"ae_mode": "auto", "night_mode": True, "metadata_quality": "video_image_only"},
            "pose": {"imu_quality": "missing"},
        }
        for index in range(1, len(frame_paths) + 1)
    ]
    tracks: list[dict[str, Any]] = []
    for track in linked_tracks:
        for detection in track.detections:
            tracks.append(
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
                    "optional_cue_scores": {"yolo_class_id": float(detection.class_id)},
                }
            )
    tracks.sort(key=lambda row: (row["frame_id"], row["track_id"]))
    return {
        "clip_id": clip_id,
        "device_id": "eval_pres_sample_video",
        "calibration_level": 1,
        "policy_id": "rbccps_measurement_policy_v1",
        "video_uri": None,
        "frames": frames,
        "tracks": tracks,
        "optional_calibration": {
            "photometric": {"field_lux_calibration_id": None},
            "map_priors": {"route_group": "eval_pres_video_demo"},
        },
    }


def write_detection_tables(out: Path, detections_by_frame: list[list[FrameDetection]], linked_tracks: list[LinkedTrack]) -> None:
    detection_rows = []
    for detections in detections_by_frame:
        for detection in detections:
            detection_rows.append({
                "frame_id": detection.frame_index,
                "bbox_xyxy": list(detection.bbox_xyxy),
                "score": detection.score,
                "class_id": detection.class_id,
            })
    track_rows = []
    track_frame_rows = []
    for track in linked_tracks:
        scores = [d.score for d in track.detections]
        frame_ids = [d.frame_index for d in track.detections]
        track_rows.append({
            "track_id": track.track_id,
            "first_frame": min(frame_ids),
            "last_frame": max(frame_ids),
            "track_length_frames": len(frame_ids),
            "mean_score": round(float(np.mean(scores)), 4),
            "max_score": round(max(scores), 4),
        })
        for detection in track.detections:
            track_frame_rows.append({
                "track_id": track.track_id,
                "frame_id": detection.frame_index,
                "bbox_xyxy": list(detection.bbox_xyxy),
                "score": detection.score,
                "class_id": detection.class_id,
            })
    write_json(out / "detections.json", detection_rows)
    write_json(out / "tracks.json", {"tracks": track_rows, "track_frames": track_frame_rows})
    write_csv(out / "detections.csv", detection_rows)
    write_csv(out / "tracks.csv", track_rows)


def run_measurement(manifest_path: Path, measurement_dir: Path):
    root = repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from rbccps_measurement.contracts.input_schema import ClipManifest
    from rbccps_measurement.ingest.validation import validate_clip_manifest
    from rbccps_measurement.pipeline import run_clip_to_directory

    manifest = ClipManifest.load(manifest_path)
    if not manifest.tracks:
        measurement_dir.mkdir(parents=True, exist_ok=True)
        write_json(measurement_dir / "reports.json", [])
        write_csv(measurement_dir / "reports.csv", [])
        write_json(measurement_dir / "reports.geojson", {"type": "FeatureCollection", "features": []})
        write_json(measurement_dir / "overlays.json", {"overlays": []})
        return []
    validate_clip_manifest(manifest)
    return run_clip_to_directory(manifest_path, measurement_dir, measurement_run_id="eval_pres_untrained_video_demo")


def render_processed_frames(manifest: dict[str, Any], reports: list[dict[str, Any]], frame_root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_by_track = {str(report.get("lamp_track_id")): report for report in reports}
    tracks_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in manifest.get("tracks", []):
        tracks_by_frame.setdefault(int(row.get("frame_id", 0)), []).append(row)
    font = _font(16)
    small = _font(12)
    output: list[Path] = []
    for frame in manifest.get("frames", []):
        frame_id = int(frame.get("frame_id", 0))
        image_path = frame_root / str(frame.get("image_uri"))
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, image.width, 54), fill=(0, 0, 0, 215))
        draw.text((10, 7), f"RBCCPS eval_pres video demo | frame {frame_id}", fill=(255, 255, 255, 255), font=font)
        draw.text((10, 31), "Detector + IoU tracks + untrained deterministic measurement outputs", fill=(255, 220, 120, 255), font=small)
        for row in tracks_by_frame.get(frame_id, []):
            track_id = str(row.get("track_id", ""))
            bbox = [float(v) for v in row.get("bbox_xyxy", [0, 0, 0, 0])]
            report = reports_by_track.get(track_id, {})
            metrics = report.get("metrics", {}) or {}
            status = report.get("status", {}) or {}
            confidence = report.get("confidence", {}) or {}
            category = str(metrics.get("overall_category", "no_report"))
            score = metrics.get("overall_useful_illumination_score")
            color = _category_color(category)
            draw.rectangle(tuple(bbox), outline=color + (255,), width=3)
            _draw_approx_affected_region(draw, bbox, image.size, color)
            label = f"{track_id} det={float(row.get('detector_score', 0.0)):.2f} {category}"
            if score is not None:
                label += f" score={float(score):.2f}"
            if status.get("label"):
                label += f" status={status.get('label')}"
            if confidence.get("action"):
                label += f" {confidence.get('action')}"
            text_y = max(58, bbox[1] - 17)
            draw.rectangle((bbox[0], text_y - 2, min(image.width, bbox[0] + 520), text_y + 16), fill=(0, 0, 0, 190))
            draw.text((bbox[0] + 3, text_y), label, fill=color + (255,), font=small)
        target = out_dir / f"processed_{frame_id:06d}.jpg"
        image.save(target, quality=90)
        output.append(target)
    return output


def _draw_approx_affected_region(draw: ImageDraw.ImageDraw, bbox: list[float], size: tuple[int, int], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    top = min(size[1] - 1, y2 + 8)
    bottom = min(size[1] - 1, top + max(80, (y2 - y1) * 4))
    half = max(45, (x2 - x1) * 3)
    polygon = [(cx, top), (max(0, cx - half), bottom), (min(size[0] - 1, cx + half), bottom)]
    draw.polygon(polygon, fill=color + (45,), outline=color + (145,))


def _category_color(category: str) -> tuple[int, int, int]:
    return {
        "adequate": (52, 211, 153),
        "marginal": (250, 204, 21),
        "poor": (251, 146, 60),
        "unknown": (248, 113, 113),
    }.get(category, (147, 197, 253))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector + tracking + measurement + eval_pres presentation artifacts on a video.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps-sample", type=float, default=3.0)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-det", type=int, default=12)
    parser.add_argument("--iou-link-threshold", type=float, default=0.30)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--ground-truth")
    parser.add_argument("--route-distance-km", type=float)
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--measurement-max-tracks", type=int, default=30, help="Cap tracks passed into the measurement block; raw detection tables still keep all detections.")
    parser.add_argument("--preset", choices=["quick", "standard", "full"], default="standard", help="Convenience defaults; explicit numeric flags override the selected preset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_preset(args)
    summary = run_video_demo(
        video=args.video,
        out=args.out,
        fps_sample=args.fps_sample,
        conf=args.conf,
        max_frames=args.max_frames,
        max_det=args.max_det,
        iou_link_threshold=args.iou_link_threshold,
        model_path=args.model_path,
        ground_truth=args.ground_truth,
        route_distance_km=args.route_distance_km,
        skip_detector=args.skip_detector,
        measurement_max_tracks=args.measurement_max_tracks,
        preset=args.preset,
    )
    print(json.dumps(summary, indent=2))


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "quick":
        args.fps_sample = 1.0 if args.fps_sample == 3.0 else args.fps_sample
        args.max_det = 8 if args.max_det == 12 else args.max_det
        args.measurement_max_tracks = 15 if args.measurement_max_tracks == 30 else args.measurement_max_tracks
    elif args.preset == "full":
        args.fps_sample = 3.0 if args.fps_sample == 3.0 else args.fps_sample
        args.max_det = 20 if args.max_det == 12 else args.max_det
        args.measurement_max_tracks = 60 if args.measurement_max_tracks == 30 else args.measurement_max_tracks
    return args


if __name__ == "__main__":
    main()
