from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rbccps_od.config.paths import ensure_dir, repo_root
from rbccps_od.config.schemas import CueWeights
from rbccps_od.io.csv_io import write_csv
from rbccps_od.io.json_io import write_json
from rbccps_od.models.yolo26 import YOLO26Detector
from rbccps_od.pipeline.advanced_runner import (
    _coerce_result,
    _result_to_detections,
    _result_to_tracks,
    _track_to_dict,
)
from rbccps_od.pipeline.multicue_stage import MultiCueFilterStage
from rbccps_od.evaluation.metrics import MetricsManager


GMC_METHODS = {"sparseOptFlow", "orb", "sift", "ecc", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO26m video inference with BoT-SORT tracking and multi-cue filtering."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--model",
        help="Optional YOLO model checkpoint path. If omitted, uses the pretrained YOLO26m base asset as a placeholder.",
    )
    parser.add_argument("--output-dir", help="Output directory. Defaults to runs/video_pipeline/<video_stem>.")
    parser.add_argument("--name", help="Run name used when --output-dir is omitted.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Detector NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth video frame.")
    parser.add_argument("--aggregation-threshold", type=float, default=0.5, help="Multi-cue acceptance threshold.")
    parser.add_argument("--disable-multicue", action="store_true", help="Keep tracks without multi-cue filtering.")
    parser.add_argument("--tracker-config", help="Optional custom Ultralytics tracker YAML.")
    parser.add_argument(
        "--gmc-method",
        choices=sorted(GMC_METHODS),
        default="sparseOptFlow",
        help="BoT-SORT global motion compensation method.",
    )
    parser.add_argument("--track-high-thresh", type=float, default=0.25)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    parser.add_argument("--new-track-thresh", type=float, default=0.25)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--with-reid", action="store_true", help="Enable BoT-SORT ReID if the environment supports it.")
    parser.add_argument("--no-save-video", action="store_true", help="Skip annotated video export.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration without inference.")
    parser.add_argument("--enable-metrics", action="store_true", help="Enable metrics evaluation using ground-truth labels.")   
    parser.add_argument("--labels-dir", help="Directory containing YOLO label txt files for evaluation.")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def default_output_dir(video_path: Path, name: str | None) -> Path:
    return repo_root() / "runs" / "video_pipeline" / (name or video_path.stem)


def resolve_model_path(model: str, *, must_exist: bool) -> Path:
    model_path = resolve_repo_path(model)
    if must_exist and not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    return model_path


def write_botsort_config(args: argparse.Namespace, output_dir: Path) -> Path:
    tracker_path = output_dir / "tracker_botsort.yaml"
    tracker_path.write_text(
        "\n".join(
            [
                "tracker_type: botsort",
                f"track_high_thresh: {args.track_high_thresh}",
                f"track_low_thresh: {args.track_low_thresh}",
                f"new_track_thresh: {args.new_track_thresh}",
                f"track_buffer: {args.track_buffer}",
                f"match_thresh: {args.match_thresh}",
                "fuse_score: True",
                f"gmc_method: {args.gmc_method}",
                "proximity_thresh: 0.5",
                "appearance_thresh: 0.8",
                f"with_reid: {str(bool(args.with_reid))}",
                "model: auto",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tracker_path


def video_metadata(video_path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError:
        return {"fps": 0.0, "frame_count": 0, "width": 0, "height": 0}

    capture = cv2.VideoCapture(str(video_path))
    try:
        return {
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        capture.release()


def _time_sec(frame_index: int, fps: float, vid_stride: int) -> float:
    if fps <= 0:
        return 0.0
    return ((frame_index - 1) * max(1, vid_stride)) / fps

def load_gt_count(
    labels_dir: Path,
    frame_index: int,
    vid_stride: int,
) -> int:

    actual_frame_index = ((frame_index - 1) * max(1, vid_stride)) + 1
    label_path = labels_dir / f"frame_{actual_frame_index:06d}.txt"

    if not label_path.exists():
        return 0

    with label_path.open("r", encoding="utf-8") as f:
        return len(f.readlines())

def _cue_values(entry: dict[str, Any]) -> dict[str, float]:
    return {cue.name: float(cue.value) for cue in entry["cues"]}


def _serialize_filter_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "track": _track_to_dict(entry["track"]),
        "aggregate_score": float(entry["aggregate_score"]),
        "accepted": bool(entry["accepted"]),
        "cues": [cue.__dict__ for cue in entry["cues"]],
    }


def _box_rows(
    result: Any,
    frame_index: int,
    time_sec: float,
    filtered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if result is None or getattr(result, "boxes", None) is None:
        return []

    boxes = result.boxes
    xywh = boxes.xywh.cpu().tolist() if getattr(boxes, "xywh", None) is not None else []
    xyxy = boxes.xyxy.cpu().tolist() if getattr(boxes, "xyxy", None) is not None else [[] for _ in xywh]
    confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xywh)
    cls_ids = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else [0] * len(xywh)
    ids = boxes.id.cpu().tolist() if getattr(boxes, "id", None) is not None else [None] * len(xywh)
    names = getattr(result, "names", {}) or {}

    rows: list[dict[str, Any]] = []
    for idx, (box, corners, conf, cls_id, track_id) in enumerate(zip(xywh, xyxy, confs, cls_ids, ids), start=1):
        track_name = f"track_{int(track_id)}" if track_id is not None else f"track_unassigned_{idx}"
        entry = next((item for item in filtered if item["track"].track_id == track_name), None)
        cues = _cue_values(entry) if entry else {}
        label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
        rows.append(
            {
                "frame_index": frame_index,
                "time_sec": f"{time_sec:.6f}",
                "track_id": track_name,
                "label": label,
                "score": f"{float(conf):.6f}",
                "x_center": f"{float(box[0]):.3f}",
                "y_center": f"{float(box[1]):.3f}",
                "width": f"{float(box[2]):.3f}",
                "height": f"{float(box[3]):.3f}",
                "x1": f"{float(corners[0]):.3f}" if len(corners) >= 4 else "",
                "y1": f"{float(corners[1]):.3f}" if len(corners) >= 4 else "",
                "x2": f"{float(corners[2]):.3f}" if len(corners) >= 4 else "",
                "y2": f"{float(corners[3]):.3f}" if len(corners) >= 4 else "",
                "aggregate_score": f"{float(entry['aggregate_score']):.6f}" if entry else "",
                "accepted": bool(entry["accepted"]) if entry else False,
                "trajectory": f"{cues.get('trajectory', 0.0):.6f}" if entry else "",
                "size_progression": f"{cues.get('size_progression', 0.0):.6f}" if entry else "",
                "light_characteristics": f"{cues.get('light_characteristics', 0.0):.6f}" if entry else "",
                "position_prior": f"{cues.get('position_prior', 0.0):.6f}" if entry else "",
                "history_len": len(entry["track"].history) if entry else 0,
            }
        )
    return rows


def _update_track_stats(
    stats: dict[str, dict[str, Any]],
    frame_index: int,
    time_sec: float,
    filtered: list[dict[str, Any]],
) -> None:
    for entry in filtered:
        track = entry["track"]
        row = stats.setdefault(
            track.track_id,
            {
                "track_id": track.track_id,
                "first_frame": frame_index,
                "first_time_sec": time_sec,
                "last_frame": frame_index,
                "last_time_sec": time_sec,
                "visible_frames": 0,
                "accepted_frames": 0,
                "scores": [],
                "aggregate_scores": [],
                "max_history_len": 0,
            },
        )
        row["last_frame"] = frame_index
        row["last_time_sec"] = time_sec
        row["visible_frames"] += 1
        row["accepted_frames"] += int(bool(entry["accepted"]))
        row["scores"].append(float(track.score))
        row["aggregate_scores"].append(float(entry["aggregate_score"]))
        row["max_history_len"] = max(int(row["max_history_len"]), len(track.history))


def _track_summary_rows(stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track_id in sorted(stats):
        item = stats[track_id]
        scores = item["scores"]
        aggregate_scores = item["aggregate_scores"]
        rows.append(
            {
                "track_id": track_id,
                "first_frame": item["first_frame"],
                "last_frame": item["last_frame"],
                "first_time_sec": f"{float(item['first_time_sec']):.6f}",
                "last_time_sec": f"{float(item['last_time_sec']):.6f}",
                "visible_frames": item["visible_frames"],
                "accepted_frames": item["accepted_frames"],
                "accepted_any": item["accepted_frames"] > 0,
                "mean_score": f"{sum(scores) / len(scores):.6f}" if scores else "0.000000",
                "max_score": f"{max(scores):.6f}" if scores else "0.000000",
                "mean_aggregate_score": f"{sum(aggregate_scores) / len(aggregate_scores):.6f}" if aggregate_scores else "0.000000",
                "max_aggregate_score": f"{max(aggregate_scores):.6f}" if aggregate_scores else "0.000000",
                "max_history_len": item["max_history_len"],
            }
        )
    return rows


def _open_video_writer(path: Path, fps: float, frame: Any) -> Any:
    import cv2

    height, width = frame.shape[:2]
    writer_fps = fps if fps > 0 else 30.0
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, writer_fps, (width, height))


def _write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    track_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    write_csv(output_dir / "frames.csv", frame_rows, list(frame_rows[0].keys()) if frame_rows else ["frame_index"])
    write_csv(
        output_dir / "detections.csv",
        detection_rows,
        list(detection_rows[0].keys()) if detection_rows else ["frame_index"],
    )
    write_csv(output_dir / "tracks.csv", track_rows, list(track_rows[0].keys()) if track_rows else ["track_id"])
    summary = {
        "config": payload,
        "video": metadata,
        "frames_processed": len(frame_rows),
        "detections_total": sum(int(row["detection_count"]) for row in frame_rows),
        "accepted_total": sum(int(row["accepted_count"]) for row in frame_rows),
        "unique_tracks": len(track_rows),
        "accepted_unique_tracks": sum(1 for row in track_rows if row["accepted_any"]),
    }
    write_json(output_dir / "summary.json", summary)


def main() -> None:
    args = parse_args()
    video_path = resolve_repo_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else default_output_dir(video_path, args.name)
    ensure_dir(output_dir)
    model_path = resolve_model_path(args.model, must_exist=not args.dry_run) if args.model else None
    tracker_path = resolve_repo_path(args.tracker_config) if args.tracker_config else write_botsort_config(args, output_dir)
    save_video = not args.no_save_video
    labels_dir = (
        resolve_repo_path(args.labels_dir)
        if args.labels_dir
        else None
    )
    if args.enable_metrics and labels_dir is None:
        raise ValueError("--labels-dir must be provided when --enable-metrics is used.")

    detector = YOLO26Detector(checkpoint_path=model_path) if model_path else YOLO26Detector()
    payload = {
        "video": str(video_path),
        "model": str(model_path) if model_path else detector.asset_name,
        "output_dir": str(output_dir),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "vid_stride": args.vid_stride,
        "tracker": str(tracker_path),
        "tracker_type": "botsort",
        "camera_motion_compensation": {"enabled": True, "gmc_method": args.gmc_method},
        "multicue": {"enabled": not args.disable_multicue, "threshold": args.aggregation_threshold},
        "save_video": save_video,
    }
    write_json(output_dir / "run_config.json", payload)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    metadata = video_metadata(video_path)
    model = detector.adapter.load(detector.resolved_checkpoint())
    multicue_stage = MultiCueFilterStage(
        CueWeights(),
        threshold=args.aggregation_threshold,
        enabled=not args.disable_multicue,
    )

    histories: dict[str, list[list[float]]] = {}
    frame_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    track_stats: dict[str, dict[str, Any]] = {}
    metrics = MetricsManager() if args.enable_metrics else None
    writer = None
    annotated_video_path = output_dir / "annotated.mp4"

    jsonl_path = output_dir / "frames.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        results = model.track(
            source=str(video_path),
            stream=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            tracker=str(tracker_path),
            persist=True,
            vid_stride=args.vid_stride,
            verbose=False,
            save=False,
        )
        for frame_index, raw_result in enumerate(results, start=1):
            result = _coerce_result(raw_result)
            time_sec = _time_sec(frame_index, float(metadata["fps"]), args.vid_stride)
            detections = _result_to_detections(result)
            tracks = _result_to_tracks(result, histories)
            frame_height = getattr(result, "orig_shape", [None, None])[0] if result is not None else None
            filtered = multicue_stage.run(tracks, frame_height=frame_height)
            accepted_tracks = [
                entry
                for entry in filtered
                if entry["accepted"]
            ]
            if args.enable_metrics and metrics is not None and labels_dir is not None:
                gt_count = load_gt_count(
                    labels_dir,
                    frame_index,
                    args.vid_stride
                )

                metrics.update_frame(
                    gt_count=gt_count,
                    accepted_tracks=accepted_tracks,
                    raw_tracks=tracks,
                )

                for entry in filtered:
                    metrics.update_track_metrics(
                        track_id=entry["track"].track_id,
                        accepted=entry["accepted"],
                    )

            frame_detection_rows = _box_rows(result, frame_index, time_sec, filtered)
            detection_rows.extend(frame_detection_rows)
            _update_track_stats(track_stats, frame_index, time_sec, filtered)

            accepted_count = sum(1 for entry in filtered if entry["accepted"])
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "time_sec": f"{time_sec:.6f}",
                    "detection_count": len(detections),
                    "track_count": len(tracks),
                    "accepted_count": accepted_count,
                    "max_detection_score": f"{max((det.score for det in detections), default=0.0):.6f}",
                    "max_aggregate_score": f"{max((float(entry['aggregate_score']) for entry in filtered), default=0.0):.6f}",
                }
            )
            jsonl.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "detections": [det.__dict__ for det in detections],
                        "tracks": [_track_to_dict(track) for track in tracks],
                        "filtered_tracks": [_serialize_filter_entry(entry) for entry in filtered],
                    }
                )
                + "\n"
            )

            if save_video and result is not None:
                plotted = result.plot()
                if writer is None:
                    writer = _open_video_writer(annotated_video_path, float(metadata["fps"]), plotted)
                writer.write(plotted)

    if writer is not None:
        writer.release()

    track_rows = _track_summary_rows(track_stats)
    if args.enable_metrics and metrics is not None:
        metrics_summary = metrics.summary()
        write_json(
            output_dir / "metrics.json",
            metrics_summary,
        )
    _write_outputs(output_dir, payload, frame_rows, detection_rows, track_rows, metadata)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "summary_json": str((output_dir / "summary.json").resolve()),
                "frames_csv": str((output_dir / "frames.csv").resolve()),
                "detections_csv": str((output_dir / "detections.csv").resolve()),
                "tracks_csv": str((output_dir / "tracks.csv").resolve()),
                "frames_jsonl": str(jsonl_path.resolve()),
                "annotated_video": str(annotated_video_path.resolve()) if save_video else None,
                "metrics_json": str((output_dir / "metrics.json").resolve()) if args.enable_metrics else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
