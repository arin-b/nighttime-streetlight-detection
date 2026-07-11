from __future__ import annotations

import argparse
from pathlib import Path

from rbccps_annotator.augment import generate_confounder_augmentations
from rbccps_annotator.bundle_workflow import prepare_bundle_workspace, validate_tutorial_examples
from rbccps_annotator.exports import export_measurement, export_yolo
from rbccps_annotator.schema import workspace_for_frames
from rbccps_annotator.segmenter import download_segmenter, status as segmenter_status
from rbccps_annotator.server import run_server
from rbccps_annotator.workspace import create_workspace_from_detector_manifest, create_workspace_from_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="RBCCPS modular streetlight and measurement annotator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_frames = subparsers.add_parser("ingest-frames", help="Create an annotation workspace from extracted frames.")
    ingest_frames.add_argument("--frames", required=True, type=Path)
    ingest_frames.add_argument("--workspace", type=Path)
    ingest_frames.add_argument("--dataset-id", default="local_frames")
    ingest_frames.add_argument("--route-id", default="")
    ingest_frames.add_argument("--clip-id", default="")
    ingest_frames.add_argument("--split", default="unassigned")
    ingest_frames.add_argument("--source-pool", default="raw_frames")

    ingest_run = subparsers.add_parser("ingest-detector-run", help="Create a workspace from a measurement/object-detector clip manifest.")
    ingest_run.add_argument("--manifest", required=True, type=Path)
    ingest_run.add_argument("--workspace", required=True, type=Path)

    serve = subparsers.add_parser("serve", help="Run the local browser annotator.")
    serve.add_argument("--workspace", required=True, type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8789)
    serve.add_argument("--no-browser", action="store_true")

    launch = subparsers.add_parser("launch", help="Ingest raw frames and launch the annotator.")
    launch.add_argument("--frames", required=True, type=Path)
    launch.add_argument("--workspace", type=Path)
    launch.add_argument("--dataset-id", default="local_frames")
    launch.add_argument("--route-id", default="")
    launch.add_argument("--clip-id", default="")
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--port", type=int, default=8789)
    launch.add_argument("--no-browser", action="store_true")

    bundle = subparsers.add_parser("prepare-bundle-workspace", help="Process portable bundle input_raw into a sampled annotation workspace.")
    bundle.add_argument("--bundle-root", type=Path, default=Path("."))
    bundle.add_argument("--input-raw", type=Path)
    bundle.add_argument("--workspace", type=Path)
    bundle.add_argument("--batch-id", default="")
    bundle.add_argument("--sample-budget", type=int)
    bundle.add_argument("--fps", type=float, default=1.0)
    bundle.add_argument("--detector-weights", type=Path)
    bundle.add_argument("--tutorial-examples", type=Path)
    bundle.add_argument("--force", action="store_true")

    bundle_launch = subparsers.add_parser("bundle-launch", help="Prepare portable bundle workspace and launch the annotator.")
    bundle_launch.add_argument("--bundle-root", type=Path, default=Path("."))
    bundle_launch.add_argument("--input-raw", type=Path)
    bundle_launch.add_argument("--workspace", type=Path)
    bundle_launch.add_argument("--batch-id", default="")
    bundle_launch.add_argument("--sample-budget", type=int)
    bundle_launch.add_argument("--fps", type=float, default=1.0)
    bundle_launch.add_argument("--detector-weights", type=Path)
    bundle_launch.add_argument("--tutorial-examples", type=Path)
    bundle_launch.add_argument("--force", action="store_true")
    bundle_launch.add_argument("--host", default="127.0.0.1")
    bundle_launch.add_argument("--port", type=int, default=8789)
    bundle_launch.add_argument("--no-browser", action="store_true")

    tutorial = subparsers.add_parser("validate-tutorial", help="Validate image+JSON tutorial examples for the portable bundle.")
    tutorial.add_argument("--tutorial-examples", required=True, type=Path)
    tutorial.add_argument("--workspace", type=Path)

    yolo = subparsers.add_parser("export-yolo", help="Export reviewed lamp-head and pole boxes as a two-class YOLO dataset.")
    yolo.add_argument("--workspace", required=True, type=Path)
    yolo.add_argument("--output", type=Path)
    yolo.add_argument("--include-candidates", action="store_true")
    yolo.add_argument("--split-dirs", action="store_true", help="Write images/labels under train/valid/test split directories.")

    measurement = subparsers.add_parser("export-measurement", help="Export measurement-block annotation tables.")
    measurement.add_argument("--workspace", required=True, type=Path)
    measurement.add_argument("--output", type=Path)

    augment = subparsers.add_parser("generate-confounder-augmentations", help="Create derived facade/confounder masked augmentation views.")
    augment.add_argument("--workspace", required=True, type=Path)
    augment.add_argument("--output", type=Path)
    augment.add_argument("--probability", type=float, default=0.2)
    augment.add_argument("--variant", choices=["dim", "blur", "gray", "noise"], default="dim")
    augment.add_argument("--seed", type=int, default=17)

    segmenter = subparsers.add_parser("download-segmenter", help="Download/cache SAM2 for Magic Surface proposals.")
    segmenter.add_argument("--engine", choices=["sam2", "fastsam", "both"], default="sam2")
    segmenter.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "ingest-frames":
        workspace = args.workspace or workspace_for_frames(args.frames)
        create_workspace_from_frames(args.frames, workspace, args.dataset_id, args.route_id, args.clip_id, args.split, args.source_pool)
        print(f"Workspace created: {workspace.resolve()}")
    elif args.command == "ingest-detector-run":
        create_workspace_from_detector_manifest(args.manifest, args.workspace)
        print(f"Workspace created: {args.workspace.resolve()}")
    elif args.command == "serve":
        run_server(args.workspace, args.host, args.port, not args.no_browser)
    elif args.command == "launch":
        workspace = args.workspace or workspace_for_frames(args.frames)
        create_workspace_from_frames(args.frames, workspace, args.dataset_id, args.route_id, args.clip_id, "unassigned", "raw_frames")
        run_server(workspace, args.host, args.port, not args.no_browser)
    elif args.command == "prepare-bundle-workspace":
        result = prepare_bundle_workspace(
            bundle_root=args.bundle_root,
            input_raw=args.input_raw,
            workspace=args.workspace,
            batch_id=args.batch_id or None,
            sample_budget=args.sample_budget,
            fps=args.fps,
            detector_weights=args.detector_weights,
            tutorial_examples=args.tutorial_examples,
            force=args.force,
        )
        print(result.to_dict())
    elif args.command == "bundle-launch":
        result = prepare_bundle_workspace(
            bundle_root=args.bundle_root,
            input_raw=args.input_raw,
            workspace=args.workspace,
            batch_id=args.batch_id or None,
            sample_budget=args.sample_budget,
            fps=args.fps,
            detector_weights=args.detector_weights,
            tutorial_examples=args.tutorial_examples,
            force=args.force,
        )
        print(result.to_dict())
        run_server(result.workspace, args.host, args.port, not args.no_browser)
    elif args.command == "validate-tutorial":
        count = validate_tutorial_examples(args.tutorial_examples, args.workspace)
        print(f"Tutorial examples valid: {count}")
    elif args.command == "export-yolo":
        output = args.output or args.workspace / "exports" / "yolo_standard"
        export_yolo(args.workspace, output, include_candidate_boxes=args.include_candidates, split_dirs=args.split_dirs)
        print(f"YOLO export written: {output.resolve()}")
    elif args.command == "export-measurement":
        output = args.output or args.workspace / "exports" / "measurement_annotations"
        export_measurement(args.workspace, output)
        print(f"Measurement export written: {output.resolve()}")
    elif args.command == "generate-confounder-augmentations":
        output = args.output or args.workspace / "exports" / f"confounder_aug_{args.variant}"
        generate_confounder_augmentations(args.workspace, output, args.probability, args.variant, args.seed)
        print(f"Augmentations written: {output.resolve()}")
    elif args.command == "download-segmenter":
        paths = download_segmenter(force=args.force, engine=args.engine)
        for path in paths:
            print(f"Prompt segmenter ready: {path.resolve()}")
        print(segmenter_status().to_dict())
