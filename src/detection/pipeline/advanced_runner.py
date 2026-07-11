from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rbccps_od.config.paths import ensure_dir, repo_root
from rbccps_od.config.schemas import AdvancedPipelineConfig, CueWeights
from rbccps_od.domain.detections import Detection
from rbccps_od.domain.tracks import Track
from rbccps_od.models.domain_adaptation import DomainAdaptationAdapter
from rbccps_od.models.enhancer import LowLightEnhancer
from rbccps_od.models.retinex import RetinexDecompositionModel
from rbccps_od.models.yolo26 import YOLO26Detector
from rbccps_od.pipeline.decomposition_stage import DecompositionStage
from rbccps_od.pipeline.detection_stage import DetectionStage
from rbccps_od.pipeline.enhancement_stage import EnhancementStage
from rbccps_od.pipeline.multicue_stage import MultiCueFilterStage
from rbccps_od.pipeline.paired_input import PairedInputFrame
from rbccps_od.pipeline.tracking_stage import TrackingStage

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the advanced nighttime streetlight pipeline.")
    parser.add_argument("--image", help="Single dark/original frame path.")
    parser.add_argument("--image-dir", help="Directory of still images for batch execution.")
    parser.add_argument("--sequence-root", help="Directory of ordered frames for sequence execution.")
    parser.add_argument("--sequence-manifest", help="Optional JSON or TXT manifest describing frame sequences.")
    parser.add_argument("--model", help="Optional explicit YOLO checkpoint path.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="advanced_pipeline")
    parser.add_argument("--output-dir", help="Optional explicit output directory.")
    parser.add_argument("--enable-enhancement", action="store_true")
    parser.add_argument("--enable-paired-input", action="store_true")
    parser.add_argument("--enable-retinex", action="store_true")
    parser.add_argument("--enable-domain-adaptation", action="store_true")
    parser.add_argument("--enable-tracking", action="store_true")
    parser.add_argument("--enable-multicue", action="store_true")
    parser.add_argument("--aggregation-threshold", type=float, default=0.5)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def infer_frame_index(path: Path) -> int:
    match = re.search(r"_frame_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    digits = re.findall(r"(\d+)", path.stem)
    return int(digits[-1]) if digits else 0


def group_sequence_paths(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(paths):
        stem = path.stem
        prefix = stem.split("__", 1)[0]
        group_name = stem
        if "__" in stem:
            suffix = stem.split("__", 1)[1]
            if "_frame_" in suffix:
                group_name = f"{prefix}__{suffix.split('_frame_', 1)[0]}"
            else:
                group_name = stem
        groups[group_name].append(path)
    for key in list(groups):
        groups[key] = sorted(groups[key], key=infer_frame_index)
    return dict(groups)


def load_sequence_manifest(path: Path) -> dict[str, list[Path]]:
    payload = path.read_text(encoding="utf-8-sig").strip()
    if not payload:
        return {}
    if path.suffix.lower() == ".json":
        data = json.loads(payload)
        if isinstance(data, dict):
            return {str(k): [Path(item).resolve() for item in v] for k, v in data.items()}
        if isinstance(data, list):
            items = [Path(item).resolve() for item in data]
            return group_sequence_paths(items)
    items = [Path(line.strip()).resolve() for line in payload.splitlines() if line.strip()]
    return group_sequence_paths(items)


def collect_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _resolve_run_mode(args: argparse.Namespace) -> str:
    provided = [bool(args.image), bool(args.image_dir), bool(args.sequence_root), bool(args.sequence_manifest)]
    if sum(provided) != 1:
        raise SystemExit("Provide exactly one of --image, --image-dir, --sequence-root, or --sequence-manifest.")
    if args.image:
        return "single_image"
    if args.image_dir:
        return "image_batch"
    return "sequence"


def _default_output_dir(name: str) -> Path:
    return repo_root() / "runs" / name


def _overlay_save(path: Path, result: Any) -> None:
    image = result.plot()
    try:
        import cv2

        cv2.imwrite(str(path), image)
    except Exception:
        from PIL import Image
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        Image.fromarray(arr).save(path)


def _coerce_result(raw_result: Any) -> Any:
    if isinstance(raw_result, (list, tuple)):
        return raw_result[0] if raw_result else None
    return raw_result


def _result_to_detections(result: Any) -> list[Detection]:
    if result is None or getattr(result, "boxes", None) is None:
        return []
    boxes = result.boxes
    xywh = boxes.xywh.cpu().tolist() if getattr(boxes, "xywh", None) is not None else []
    confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xywh)
    cls_ids = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else [0] * len(xywh)
    xyxy = boxes.xyxy.cpu().tolist() if getattr(boxes, "xyxy", None) is not None else [[] for _ in xywh]
    names = getattr(result, "names", {}) or {}
    detections: list[Detection] = []
    for box, conf, cls_id, corners in zip(xywh, confs, cls_ids, xyxy):
        label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
        detections.append(
            Detection(
                bbox=[float(v) for v in box],
                score=float(conf),
                label=label,
                source="detector",
                metadata={
                    "xyxy": [float(v) for v in corners],
                    "orig_shape": list(getattr(result, "orig_shape", [])),
                    "path": str(getattr(result, "path", "")),
                },
            )
        )
    return detections


def _result_to_tracks(result: Any, histories: dict[str, list[list[float]]]) -> list[Track]:
    if result is None or getattr(result, "boxes", None) is None:
        return []
    boxes = result.boxes
    xywh = boxes.xywh.cpu().tolist() if getattr(boxes, "xywh", None) is not None else []
    confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xywh)
    ids = boxes.id.cpu().tolist() if getattr(boxes, "id", None) is not None else [None] * len(xywh)
    tracks: list[Track] = []
    for idx, (box, conf, track_id) in enumerate(zip(xywh, confs, ids), start=1):
        track_name = f"track_{int(track_id)}" if track_id is not None else f"track_unassigned_{idx}"
        history = histories.setdefault(track_name, [])
        history.append([float(v) for v in box])
        tracks.append(
            Track(
                track_id=track_name,
                bbox=[float(v) for v in box],
                score=float(conf),
                age=max(0, len(history) - 1),
                hits=len(history),
                history=list(history),
                metadata={"result_path": str(getattr(result, "path", ""))},
            )
        )
    return tracks


def _track_to_dict(track: Track) -> dict[str, Any]:
    return {
        "track_id": track.track_id,
        "bbox": track.bbox,
        "score": track.score,
        "age": track.age,
        "hits": track.hits,
        "history": track.history,
        "metadata": track.metadata,
    }


def _build_config(args: argparse.Namespace) -> AdvancedPipelineConfig:
    return AdvancedPipelineConfig(
        enable_enhancement=args.enable_enhancement,
        enable_paired_input=args.enable_paired_input,
        enable_retinex=args.enable_retinex,
        enable_domain_adaptation=args.enable_domain_adaptation,
        enable_tracking=args.enable_tracking,
        enable_multicue=args.enable_multicue,
        aggregation_threshold=args.aggregation_threshold,
        cue_weights=CueWeights(),
    )


def _build_payload(args: argparse.Namespace, config: AdvancedPipelineConfig, run_mode: str, output_dir: Path) -> dict[str, Any]:
    return {
        "mode": run_mode,
        "image": str(Path(args.image).resolve()) if args.image else None,
        "image_dir": str(Path(args.image_dir).resolve()) if args.image_dir else None,
        "sequence_root": str(Path(args.sequence_root).resolve()) if args.sequence_root else None,
        "sequence_manifest": str(Path(args.sequence_manifest).resolve()) if args.sequence_manifest else None,
        "model": str(Path(args.model).resolve()) if args.model else None,
        "output_dir": str(output_dir.resolve()),
        "config": config.__dict__ | {"cue_weights": config.cue_weights.__dict__},
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "tracker": args.tracker,
    }


def _process_still_frame(
    image_path: Path,
    detector: YOLO26Detector,
    detection_stage: DetectionStage,
    enhancement: EnhancementStage,
    decomposition: DecompositionStage,
    tracking_stage: TrackingStage,
    multicue_stage: MultiCueFilterStage,
    output_dir: Path,
    args: argparse.Namespace,
    run_mode: str,
    detector_model: Any | None = None,
) -> dict[str, Any]:
    overlay_dir = ensure_dir(output_dir / "overlays")
    detection_dir = ensure_dir(output_dir / "detections")
    enhanced_dir = ensure_dir(output_dir / "enhanced")
    retinex_dir = ensure_dir(output_dir / "retinex")

    enhanced_path = enhancement.run(
        image_path,
        output_path=enhanced_dir / image_path.name if args.enable_enhancement else None,
        device=args.device,
    )
    paired = PairedInputFrame(dark_frame=image_path, light_frame=Path(enhanced_path) if args.enable_paired_input else None)
    components = decomposition.run(
        enhanced_path,
        output_dir=retinex_dir / image_path.stem if args.enable_retinex else None,
        device=args.device,
    )
    detector_input = components.get("reflectance", enhanced_path)
    domain_adaptation_enabled = getattr(detection_stage, "enable_domain_adaptation", False)
    if detector_model is None or domain_adaptation_enabled:
        raw_result = detection_stage.run(detector_input, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
    else:
        raw_result = detector_model.predict(source=str(detector_input), conf=args.conf, iou=args.iou, device=args.device, verbose=False)
    result = _coerce_result(raw_result)
    detections = _result_to_detections(result)
    tracks = tracking_stage.run(detections)
    filtered = multicue_stage.run(tracks, frame_height=(getattr(result, "orig_shape", [None, None])[0] if result is not None else None))
    if result is not None:
        _overlay_save(overlay_dir / image_path.name, result)
    item = {
        "mode": run_mode,
        "input_image": str(image_path.resolve()),
        "paired_input": {
            "dark_frame": str(paired.dark_frame.resolve()),
            "light_frame": str(paired.light_frame.resolve()) if paired.light_frame else None,
        },
        "enhanced_image": enhanced_path,
        "decomposition": components,
        "detections": [det.__dict__ for det in detections],
        "tracks": [_track_to_dict(track) for track in tracks],
        "filtered_tracks": [
            {
                "track": _track_to_dict(item["track"]),
                "aggregate_score": item["aggregate_score"],
                "accepted": item["accepted"],
                "cues": [cue.__dict__ for cue in item["cues"]],
            }
            for item in filtered
        ],
    }
    with (detection_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(item, handle, indent=2)
    return {
        "mode": run_mode,
        "group": image_path.parent.name,
        "image": image_path.name,
        "input_image": str(image_path.resolve()),
        "detector_input": str(Path(detector_input).resolve()),
        "detection_count": len(detections),
        "accepted_count": sum(1 for entry in filtered if entry["accepted"]),
        "track_count": len(tracks),
        "max_score": max((det.score for det in detections), default=0.0),
    }


def _process_sequence_group(
    group_name: str,
    frames: list[Path],
    detector: YOLO26Detector,
    enhancement: EnhancementStage,
    decomposition: DecompositionStage,
    multicue_stage: MultiCueFilterStage,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    overlay_dir = ensure_dir(output_dir / "overlays" / group_name)
    detection_dir = ensure_dir(output_dir / "detections" / group_name)
    enhanced_dir = ensure_dir(output_dir / "enhanced" / group_name)
    retinex_dir = ensure_dir(output_dir / "retinex" / group_name)
    raw_dir = ensure_dir(output_dir / "raw_tracks" / group_name)
    detector_model = detector.adapter.load(detector.resolved_checkpoint())
    track_histories: dict[str, list[list[float]]] = {}
    rows: list[dict[str, Any]] = []
    for frame in frames:
        enhanced_path = enhancement.run(frame, output_path=enhanced_dir / frame.name if args.enable_enhancement else None, device=args.device)
        components = decomposition.run(
            enhanced_path,
            output_dir=retinex_dir / frame.stem if args.enable_retinex else None,
            device=args.device,
        )
        detector_input = components.get("reflectance", enhanced_path)
        if args.enable_tracking:
            raw_result = detector_model.track(
                source=str(detector_input),
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                tracker=args.tracker,
                persist=True,
                verbose=False,
            )
        else:
            raw_result = detector_model.predict(
                source=str(detector_input),
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
        result = _coerce_result(raw_result)
        detections = _result_to_detections(result)
        if args.enable_tracking:
            tracks = _result_to_tracks(result, track_histories)
        else:
            tracks = [Track(track_id=f"{frame.stem}_det_{idx}", bbox=det.bbox, score=det.score, history=[det.bbox]) for idx, det in enumerate(detections, start=1)]
        filtered = multicue_stage.run(tracks, frame_height=(getattr(result, "orig_shape", [None, None])[0] if result is not None else None))
        if result is not None:
            _overlay_save(overlay_dir / frame.name, result)
        item = {
            "mode": "sequence",
            "group": group_name,
            "input_image": str(frame.resolve()),
            "enhanced_image": enhanced_path,
            "decomposition": components,
            "detections": [det.__dict__ for det in detections],
            "tracks": [_track_to_dict(track) for track in tracks],
            "filtered_tracks": [
                {
                    "track": _track_to_dict(entry["track"]),
                    "aggregate_score": entry["aggregate_score"],
                    "accepted": entry["accepted"],
                    "cues": [cue.__dict__ for cue in entry["cues"]],
                }
                for entry in filtered
            ],
        }
        with (detection_dir / f"{frame.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(item, handle, indent=2)
        with (raw_dir / f"{frame.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump([_track_to_dict(track) for track in tracks], handle, indent=2)
        rows.append(
            {
                "mode": "sequence",
                "group": group_name,
                "image": frame.name,
                "input_image": str(frame.resolve()),
                "detector_input": str(Path(detector_input).resolve()),
                "detection_count": len(detections),
                "accepted_count": sum(1 for entry in filtered if entry["accepted"]),
                "track_count": len(tracks),
                "max_score": max((det.score for det in detections), default=0.0),
            }
        )
    return rows


def _write_summary(output_dir: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    if rows:
        fieldnames = list(rows[0].keys())
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "config": payload,
        "images_total": len(rows),
        "detections_total": sum(int(row["detection_count"]) for row in rows),
        "accepted_total": sum(int(row["accepted_count"]) for row in rows),
        "track_total": sum(int(row["track_count"]) for row in rows),
        "groups": sorted({row["group"] for row in rows}),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    run_mode = _resolve_run_mode(args)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(args.name)
    config = _build_config(args)
    payload = _build_payload(args, config, run_mode, output_dir)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    enhancement = EnhancementStage(
        LowLightEnhancer(enabled=config.enable_enhancement, asset_name=config.enhancement_asset),
        enabled=config.enable_enhancement,
    )
    decomposition = DecompositionStage(
        RetinexDecompositionModel(enabled=config.enable_retinex, asset_name=config.retinex_asset),
        enabled=config.enable_retinex,
    )
    detector = YOLO26Detector(checkpoint_path=args.model) if args.model else YOLO26Detector(asset_name=config.detector_asset)
    detection_stage = DetectionStage(
        detector=detector,
        domain_adapter=DomainAdaptationAdapter(enabled=config.enable_domain_adaptation),
        enable_domain_adaptation=config.enable_domain_adaptation,
    )
    tracking_stage = TrackingStage(enabled=False)
    multicue_stage = MultiCueFilterStage(config.cue_weights, threshold=config.aggregation_threshold, enabled=config.enable_multicue)
    rows: list[dict[str, Any]] = []

    if run_mode == "single_image":
        rows.append(_process_still_frame(Path(args.image).resolve(), detector, detection_stage, enhancement, decomposition, tracking_stage, multicue_stage, output_dir, args, run_mode))
    elif run_mode == "image_batch":
        detector_model = detector.adapter.load(detector.resolved_checkpoint())
        for image_path in collect_images(Path(args.image_dir).resolve()):
            rows.append(_process_still_frame(image_path, detector, detection_stage, enhancement, decomposition, tracking_stage, multicue_stage, output_dir, args, run_mode, detector_model=detector_model))
    else:
        if args.sequence_manifest:
            groups = load_sequence_manifest(Path(args.sequence_manifest).resolve())
        else:
            groups = group_sequence_paths(collect_images(Path(args.sequence_root).resolve()))
        for group_name, frames in groups.items():
            rows.extend(_process_sequence_group(group_name, frames, detector, enhancement, decomposition, multicue_stage, output_dir, args))

    _write_summary(output_dir, payload, rows)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "summary_json": str((output_dir / 'summary.json').resolve()), "frames_processed": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
