from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clean import clean_eval_pres
from .cli import evaluate_to_directory
from .detection_tracking_artifacts import build_detection_tracking_artifacts
from .doctor import run_doctor
from .measurement_artifacts import build_measurement_artifacts
from .video_demo import DEFAULT_MODEL, DEFAULT_VIDEO, apply_preset, run_video_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RBCCPS evaluation presentation suite.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an existing clip manifest and reports.json.")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--reports", required=True)
    evaluate.add_argument("--ground-truth")
    evaluate.add_argument("--route-distance-km", type=float)
    evaluate.add_argument("--latency-seconds", type=float)
    evaluate.add_argument("--model-path", action="append", default=[])
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--run-name", default="raw_pretrained_coco_untrained_eval")

    demo = subparsers.add_parser("demo-video", help="Run video extraction, detection/tracking, measurement, overlays, and evaluation.")
    demo.add_argument("--video", default=str(DEFAULT_VIDEO))
    demo.add_argument("--out", required=True)
    demo.add_argument("--preset", choices=["quick", "standard", "full"], default="standard")
    demo.add_argument("--fps-sample", type=float, default=3.0)
    demo.add_argument("--conf", type=float, default=0.05)
    demo.add_argument("--max-frames", type=int)
    demo.add_argument("--max-det", type=int, default=12)
    demo.add_argument("--iou-link-threshold", type=float, default=0.30)
    demo.add_argument("--model-path", default=str(DEFAULT_MODEL))
    demo.add_argument("--ground-truth")
    demo.add_argument("--route-distance-km", type=float)
    demo.add_argument("--skip-detector", action="store_true")
    demo.add_argument("--measurement-max-tracks", type=int, default=30)

    artifacts = subparsers.add_parser("artifacts", help="Build detection/tracking or measurement artifact packs.")
    artifact_subparsers = artifacts.add_subparsers(dest="artifact_kind", required=True)
    measurement = artifact_subparsers.add_parser("measurement", help="Build measurement tables, plots, and per-lamp cards.")
    measurement.add_argument("--reports", required=True)
    measurement.add_argument("--out", required=True)
    measurement.add_argument("--manifest")
    measurement.add_argument("--frame-root")
    measurement.add_argument("--measurement-dir")
    detection = artifact_subparsers.add_parser("detection", help="Build detection/tracking tables, plots, overlays, and track cards.")
    detection.add_argument("--manifest", required=True)
    detection.add_argument("--out", required=True)
    detection.add_argument("--frame-root")
    detection.add_argument("--ground-truth")
    detection.add_argument("--video-fps", type=float, default=3.0)
    detection.add_argument("--no-overlays", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Check dependencies, default assets, and write access.")
    doctor.add_argument("--out")

    clean = subparsers.add_parser("clean", help="Remove eval_pres cache directories.")
    clean.add_argument("--yes", action="store_true", help="Actually delete listed cache directories.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "evaluate":
        summary = evaluate_to_directory(
            manifest=args.manifest,
            reports=args.reports,
            out=args.out,
            ground_truth=args.ground_truth,
            route_distance_km=args.route_distance_km,
            latency_seconds=args.latency_seconds,
            model_paths=args.model_path,
            run_name=args.run_name,
        )
        payload = {"out": str(Path(args.out)), "computed_metrics": sum(m["status"] == "computed" for m in summary["metrics"]), "plots": summary["plots"]}
    elif args.command == "demo-video":
        apply_preset(args)
        payload = run_video_demo(
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
    elif args.command == "artifacts" and args.artifact_kind == "measurement":
        payload = build_measurement_artifacts(
            reports_path=args.reports,
            out_dir=args.out,
            manifest_path=args.manifest,
            frame_root=args.frame_root,
            measurement_dir=args.measurement_dir,
        )
    elif args.command == "artifacts" and args.artifact_kind == "detection":
        payload = build_detection_tracking_artifacts(
            manifest_path=args.manifest,
            out_dir=args.out,
            frame_root=args.frame_root,
            ground_truth_path=args.ground_truth,
            render_overlays=not args.no_overlays,
            video_fps=args.video_fps,
        )
    elif args.command == "doctor":
        payload = run_doctor(args.out)
    elif args.command == "clean":
        payload = clean_eval_pres(yes=args.yes)
    else:
        parser.error("unsupported command")
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
