"""Main orchestrator — single entry point for the audit pipeline.

Usage:
    python -m audit_pipeline.run_audit \
        --video /path/to/video.mp4 \
        --model /path/to/best.pt \
        [--gt-labels /path/to/labels/] \
        [--output-dir /path/to/output/] \
        [options]

This script ties together every component from the PDF proposal:
  1. YOLO26 detection + BoT-SORT / ByteTrack tracking
  2. Multi-cue filtering (aspect-ratio, spatial, brightness, temporal, duplicate)
  3. Brightness measurement engine
  4. Temporal aggregation
  5. Evaluation metrics (detection mAP, status classification, audit counts)
  6. Audit report generation (JSON + CSV + Markdown)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np

from audit_pipeline.config import (
    AggregationConfig,
    AuditPipelineConfig,
    DEFAULT_STREETLIGHT_TARGET_LABELS,
    DetectorConfig,
    EvaluationConfig,
    LocationPriorConfig,
    MeasurementConfig,
    MultiCueConfig,
    TrackerConfig,
)
from audit_pipeline.detector import load_model, resolve_target_classes, stream_video_with_tracking
from audit_pipeline.tracker import write_tracker_config
from audit_pipeline.multicue_filter import (
    FilterResult,
    duplicate_removal,
    filter_frame_detections,
    temporal_consistency_filter,
)
from audit_pipeline.measurement import LampMeasurement, measure_lamp
from audit_pipeline.aggregator import AggregatedLamp, aggregate_measurements
from audit_pipeline.evaluator import (
    DetectionMetrics,
    StatusClassificationMetrics,
    compute_map,
    evaluate_frame_detections,
    load_gt_boxes_yolo,
)
from audit_pipeline.report_generator import (
    write_csv_report,
    write_json_report,
    write_markdown_report,
)
from audit_pipeline.location_prior import (
    IMPLEMENTATION as LOCATION_PRIOR_IMPLEMENTATION,
    LocationPriorSettings,
    LocationPriorStore,
    build_location_prior_report,
    evidence_for_lamp,
    load_location_samples,
    static_location_sample,
    write_location_prior_report,
)


# ================================================================== #
# CLI argument parsing                                                #
# ================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streetlight Audit Pipeline — comprehensive detection, "
                    "tracking, measurement, and audit report generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ─────────────────────────────────────────────────────
    p.add_argument("--video",
                   help="Path to the input video.")
    p.add_argument("--model",
                   help="Path to YOLO fine-tuned weights (.pt).")

    # ── Output ───────────────────────────────────────────────────────
    p.add_argument("--output-dir",
                   help="Output directory. Default: runs/audit/<video_stem>")

    # ── Detector ─────────────────────────────────────────────────────
    p.add_argument("--conf", type=float, default=0.5,
                   help="YOLO confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Inference image size.")
    p.add_argument("--device", default="0",
                   help="Inference device (e.g. '0' for GPU, 'cpu').")
    p.add_argument("--classes", type=int, nargs="+",
                   help="Streetlight class indices to detect. If omitted, class IDs "
                        "are resolved from model names such as 'streetlight'.")
    p.add_argument("--target-labels", nargs="+",
                   default=list(DEFAULT_STREETLIGHT_TARGET_LABELS),
                   help="Class names that should be treated as streetlights when "
                        "--classes is omitted.")
    p.add_argument("--use-geometry-attention", action="store_true",
                   help="Patch the detector with geometry-aware vertical/horizontal attention.")
    p.add_argument("--use-cse", action="store_true",
                   help="Patch YOLO CSP/C2f blocks with the existing CSE module.")
    p.add_argument("--use-negative-attention", action="store_true",
                   help="Patch the detector with the learned negative-attention suppression branch.")

    # ── Tracker ──────────────────────────────────────────────────────
    p.add_argument("--tracker", default="botsort",
                   choices=["botsort", "bytetrack"],
                   help="Tracker type.")
    p.add_argument("--track-buffer", type=int, default=30,
                   help="Tracker lost-track buffer (frames).")
    p.add_argument("--vid-stride", type=int, default=1,
                   help="Process every Nth frame.")
    p.add_argument("--gmc-method", default="sparseOptFlow",
                   help="BoT-SORT global motion compensation method.")
    p.add_argument("--with-reid", action="store_true",
                   help="Enable ReID for BoT-SORT.")

    # ── Multi-cue filters ────────────────────────────────────────────
    p.add_argument("--disable-multicue", action="store_true",
                   help="Disable multi-cue filtering entirely.")
    p.add_argument("--aggregation-threshold", type=float, default=0.5,
                   help="Weighted multi-cue acceptance threshold.")
    p.add_argument("--aspect-ratio-min", type=float, default=0.3)
    p.add_argument("--aspect-ratio-max", type=float, default=3.0)
    p.add_argument("--spatial-upper-fraction", type=float, default=0.85)
    p.add_argument("--min-brightness-det", type=int, default=30,
                   help="Min max-pixel brightness in ROI to keep a detection.")
    p.add_argument("--min-track-confirm-frames", type=int, default=3,
                   help="Min frames for a track to be temporally confirmed.")
    p.add_argument("--duplicate-distance-px", type=float, default=50.0,
                   help="Max pixel distance to merge duplicate tracks.")

    # ── Measurement ──────────────────────────────────────────────────
    p.add_argument("--brightness-threshold", type=int, default=80,
                   help="Grayscale brightness threshold for on/off.")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Gamma correction (1.0=off, 2.2=sRGB).")
    p.add_argument("--use-hsv", action="store_true",
                   help="Use HSV V-channel instead of grayscale.")

    # ── Aggregation ──────────────────────────────────────────────────
    p.add_argument("--min-track-frames", type=int, default=5,
                   help="Discard tracks shorter than this.")
    p.add_argument("--working-fraction", type=float, default=0.75,
                   help="Fraction of 'on' frames to classify as working.")

    # ── Evaluation ───────────────────────────────────────────────────
    p.add_argument("--gt-labels",
                   help="Dir with YOLO label .txt files for evaluation.")
    p.add_argument("--gt-status-file",
                   help="CSV with (lamp_id, status) ground truth.")

    # ── Location prior / audit memory ────────────────────────────────
    p.add_argument("--location-prior",
                   help="Existing known-lamp prior JSON, or measurement-block route_aggregation.json, to query/update.")
    p.add_argument("--write-location-prior",
                   help="Where to write the updated known-lamp prior. Default: <output-dir>/known_lamp_prior.json")
    p.add_argument("--location-samples",
                   help="CSV/JSON frame telemetry with frame_index, latitude, longitude, gps_accuracy_m, device_id, route_group.")
    p.add_argument("--capture-device-id",
                   help="Device ID for this audit pass when telemetry omits it.")
    p.add_argument("--route-group",
                   help="Route or area group for this audit pass.")
    p.add_argument("--capture-lat", type=float,
                   help="Static latitude fallback for the whole capture when per-frame telemetry is unavailable.")
    p.add_argument("--capture-lon", type=float,
                   help="Static longitude fallback for the whole capture when per-frame telemetry is unavailable.")
    p.add_argument("--capture-gps-accuracy-m", type=float,
                   help="GPS accuracy for --capture-lat/--capture-lon.")
    p.add_argument("--prior-query-lat", type=float,
                   help="Latitude to query against the location prior before running detection.")
    p.add_argument("--prior-query-lon", type=float,
                   help="Longitude to query against the location prior before running detection.")
    p.add_argument("--prior-query-gps-accuracy-m", type=float,
                   help="GPS accuracy for --prior-query-lat/--prior-query-lon.")
    p.add_argument("--prior-only", action="store_true",
                   help="Only query the location prior; skip video detection and measurement.")
    p.add_argument("--prior-match-radius-m", type=float, default=12.0,
                   help="GPS merge/query radius for location-prior candidates.")
    p.add_argument("--prior-good-gps-match-radius-m", type=float, default=8.0,
                   help="Stricter merge/query radius when both positions have good GPS.")
    p.add_argument("--prior-min-observations", type=int, default=2,
                   help="Observations needed before the prior can claim that a lamp likely exists.")
    p.add_argument("--prior-min-devices", type=int, default=2,
                   help="Distinct devices needed to strengthen the known-lamp claim.")
    p.add_argument("--prior-existence-threshold", type=float, default=0.72,
                   help="Confidence threshold for known_lamp_likely_exists.")

    # ── Misc ─────────────────────────────────────────────────────────
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config and exit.")
    p.add_argument("--no-save-video", action="store_true",
                   help="Skip annotated video export.")

    return p.parse_args()


# ================================================================== #
# Config builder                                                      #
# ================================================================== #

def build_config(args: argparse.Namespace) -> AuditPipelineConfig:
    """Build the full pipeline config from CLI args."""
    return AuditPipelineConfig(
        video_path=str(Path(args.video).resolve()) if args.video else "",
        output_dir=args.output_dir,
        detector=DetectorConfig(
            model_path=str(Path(args.model).resolve()) if args.model else "",
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            target_classes=args.classes,
            target_labels=args.target_labels,
            use_geometry_attention=args.use_geometry_attention,
            use_cse=args.use_cse,
            use_negative_attention=args.use_negative_attention,
        ),
        tracker=TrackerConfig(
            tracker_type=args.tracker,
            track_buffer=args.track_buffer,
            vid_stride=args.vid_stride,
            gmc_method=args.gmc_method,
            with_reid=args.with_reid,
        ),
        multicue=MultiCueConfig(
            enabled=not args.disable_multicue,
            aggregation_threshold=args.aggregation_threshold,
            aspect_ratio_min=args.aspect_ratio_min,
            aspect_ratio_max=args.aspect_ratio_max,
            spatial_upper_fraction=args.spatial_upper_fraction,
            min_brightness_for_detection=args.min_brightness_det,
            min_track_frames_for_confirmation=args.min_track_confirm_frames,
            duplicate_center_distance_px=args.duplicate_distance_px,
        ),
        measurement=MeasurementConfig(
            brightness_threshold=args.brightness_threshold,
            gamma_correction=args.gamma,
            use_hsv_v_channel=args.use_hsv,
        ),
        aggregation=AggregationConfig(
            min_track_frames=args.min_track_frames,
            working_frame_fraction=args.working_fraction,
        ),
        evaluation=EvaluationConfig(
            gt_labels_dir=args.gt_labels,
            gt_status_file=args.gt_status_file,
        ),
        location_prior=LocationPriorConfig(
            prior_path=args.location_prior,
            output_path=args.write_location_prior,
            location_samples_path=args.location_samples,
            capture_device_id=args.capture_device_id,
            route_group=args.route_group,
            capture_latitude=args.capture_lat,
            capture_longitude=args.capture_lon,
            capture_gps_accuracy_m=args.capture_gps_accuracy_m,
            query_latitude=args.prior_query_lat,
            query_longitude=args.prior_query_lon,
            query_gps_accuracy_m=args.prior_query_gps_accuracy_m,
            prior_only=args.prior_only,
            match_radius_m=args.prior_match_radius_m,
            good_gps_match_radius_m=args.prior_good_gps_match_radius_m,
            min_observations_for_existing=args.prior_min_observations,
            min_devices_for_high_confidence=args.prior_min_devices,
            existence_confidence_threshold=args.prior_existence_threshold,
        ),
    )


# ================================================================== #
# Video metadata                                                      #
# ================================================================== #

def get_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """Extract FPS, resolution, frame count from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


# ================================================================== #
# Location prior helpers                                              #
# ================================================================== #

def _location_prior_settings(cfg: LocationPriorConfig) -> LocationPriorSettings:
    return LocationPriorSettings(
        match_radius_m=cfg.match_radius_m,
        good_gps_match_radius_m=cfg.good_gps_match_radius_m,
        min_observations_for_existing=cfg.min_observations_for_existing,
        min_devices_for_high_confidence=cfg.min_devices_for_high_confidence,
        existence_confidence_threshold=cfg.existence_confidence_threshold,
    )


def _location_prior_requested(cfg: LocationPriorConfig) -> bool:
    return any(
        [
            cfg.prior_path,
            cfg.output_path,
            cfg.location_samples_path,
            cfg.capture_latitude is not None and cfg.capture_longitude is not None,
            cfg.query_latitude is not None and cfg.query_longitude is not None,
            cfg.prior_only,
        ]
    )


def _query_payload(cfg: LocationPriorConfig) -> dict[str, Any] | None:
    if cfg.query_latitude is None or cfg.query_longitude is None:
        return None
    return {
        "latitude": cfg.query_latitude,
        "longitude": cfg.query_longitude,
        "gps_accuracy_m": cfg.query_gps_accuracy_m,
    }


def _default_output_dir(cfg: AuditPipelineConfig, fallback_name: str) -> Path:
    if cfg.output_dir:
        return Path(cfg.output_dir)
    return Path(__file__).resolve().parents[1] / "runs" / "audit" / fallback_name


def _process_location_prior(
    lamps: list[AggregatedLamp],
    cfg: AuditPipelineConfig,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any] | None:
    prior_cfg = cfg.location_prior
    if not _location_prior_requested(prior_cfg):
        return None

    settings = _location_prior_settings(prior_cfg)
    store = LocationPriorStore.load(prior_cfg.prior_path)
    query = _query_payload(prior_cfg)
    query_match = (
        store.best_match(
            float(query["latitude"]),
            float(query["longitude"]),
            query.get("gps_accuracy_m"),
            settings,
        )
        if query
        else None
    )

    samples = load_location_samples(prior_cfg.location_samples_path)
    samples.extend(
        static_location_sample(
            prior_cfg.capture_latitude,
            prior_cfg.capture_longitude,
            prior_cfg.capture_gps_accuracy_m,
            prior_cfg.capture_device_id,
            prior_cfg.route_group,
        )
    )

    lamp_updates: list[dict[str, Any]] = []
    for lamp in lamps:
        evidence = evidence_for_lamp(
            lamp,
            samples,
            run_id=run_id,
            default_device_id=prior_cfg.capture_device_id,
            default_route_group=prior_cfg.route_group,
        )
        if evidence is None:
            continue

        candidate, match, new_candidate = store.update_with_observation(evidence, settings)
        refreshed_match = match or store.best_match(evidence.latitude, evidence.longitude, evidence.gps_accuracy_m, settings)
        lamp.location = evidence.to_dict()
        lamp.existence_prior = {
            "candidate_lamp_id": candidate.candidate_lamp_id,
            "new_candidate": new_candidate,
            "match": refreshed_match.to_dict() if refreshed_match else None,
            "source": LOCATION_PRIOR_IMPLEMENTATION,
        }
        lamp_updates.append(
            {
                "lamp_track_id": lamp.track_id,
                "candidate_lamp_id": candidate.candidate_lamp_id,
                "new_candidate": new_candidate,
                "location": evidence.to_dict(),
                "match": refreshed_match.to_dict() if refreshed_match else None,
            }
        )

    updated_prior_path: Path | None = None
    if lamp_updates or prior_cfg.output_path:
        updated_prior_path = Path(prior_cfg.output_path) if prior_cfg.output_path else output_dir / "known_lamp_prior.json"
        store.write(updated_prior_path)

    report = build_location_prior_report(
        prior_path=prior_cfg.prior_path,
        updated_prior_path=str(updated_prior_path) if updated_prior_path else None,
        query=query,
        query_match=query_match,
        lamp_updates=lamp_updates,
        store=store,
        settings=settings,
    )
    report_path = write_location_prior_report(output_dir, report)
    report["report_path"] = str(report_path)
    return report


def _run_prior_only(cfg: AuditPipelineConfig) -> None:
    prior_cfg = cfg.location_prior
    query = _query_payload(prior_cfg)
    if query is None:
        print("ERROR: --prior-only requires --prior-query-lat and --prior-query-lon", file=sys.stderr)
        sys.exit(2)

    output_dir = _default_output_dir(cfg, "location_prior_query")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _location_prior_settings(prior_cfg)
    store = LocationPriorStore.load(prior_cfg.prior_path)
    query_match = store.best_match(
        float(query["latitude"]),
        float(query["longitude"]),
        query.get("gps_accuracy_m"),
        settings,
    )
    report = build_location_prior_report(
        prior_path=prior_cfg.prior_path,
        updated_prior_path=None,
        query=query,
        query_match=query_match,
        lamp_updates=[],
        store=store,
        settings=settings,
    )
    report_path = write_location_prior_report(output_dir, report)
    report["report_path"] = str(report_path)
    (output_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")
    (output_dir / "audit_report.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "prior_only": True,
                    "claim": report["query"]["claim"],
                    "known_lamp_match": report["query"]["match"],
                },
                "location_prior": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=" * 70)
    print("  LOCATION PRIOR QUERY")
    print("=" * 70)
    print(f"  Claim: {report['query']['claim']}")
    if query_match:
        print(f"  Candidate: {query_match.candidate_lamp_id}")
        print(f"  Confidence: {query_match.confidence:.3f}")
        print(f"  Distance: {query_match.distance_m:.2f} m")
    print(f"  Report: {report_path}")


# ================================================================== #
# Result extraction helpers                                           #
# ================================================================== #

class Detection(TypedDict):
    track_id: str
    xyxy: list[float]
    confidence: float
    class_id: int | None
    class_label: str


def _class_name_from_result(result: Any, class_id: int | None) -> str:
    if class_id is None:
        return "unknown"
    names = getattr(result, "names", {}) or {}
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def extract_detections(result: Any) -> list[Detection]:
    """Pull class, track_id, xyxy, and confidence from a YOLO result."""
    if result is None or getattr(result, "boxes", None) is None:
        return []

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
    confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(xyxy)
    ids = boxes.id.cpu().tolist() if getattr(boxes, "id", None) is not None else [None] * len(xyxy)
    cls_ids = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else [None] * len(xyxy)

    detections = []
    for idx, (corners, conf, track_id, cls_id) in enumerate(zip(xyxy, confs, ids, cls_ids), start=1):
        class_id = int(cls_id) if cls_id is not None else None
        tid = f"track_{int(track_id)}" if track_id is not None else f"track_unassigned_{idx}"
        detections.append({
            "track_id": tid,
            "xyxy": [float(v) for v in corners],
            "confidence": float(conf),
            "class_id": class_id,
            "class_label": _class_name_from_result(result, class_id),
        })
    return detections


def get_frame_from_result(result: Any) -> np.ndarray | None:
    """Extract the original frame from a YOLO result object."""
    if result is None:
        return None
    orig = getattr(result, "orig_img", None)
    if orig is not None:
        return orig
    return None


def keep_target_class_detections(
    detections: list[Detection],
    allowed_class_ids: set[int],
) -> tuple[list[Detection], int]:
    """Return target-class detections and number of non-target boxes dropped."""
    kept: list[Detection] = []
    suppressed = 0
    for det in detections:
        class_id = det["class_id"]
        if class_id is not None and class_id in allowed_class_ids:
            kept.append(det)
        else:
            suppressed += 1
    return kept, suppressed


def _box_to_ints(xyxy: list[float], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1 = max(0, min(w - 1, int(round(xyxy[0]))))
    y1 = max(0, min(h - 1, int(round(xyxy[1]))))
    x2 = max(0, min(w - 1, int(round(xyxy[2]))))
    y2 = max(0, min(h - 1, int(round(xyxy[3]))))
    return x1, y1, x2, y2


def draw_streetlight_annotations(
    frame: np.ndarray,
    detections: list[Detection],
    measurements: dict[str, LampMeasurement],
) -> np.ndarray:
    """Draw only audit-accepted streetlight detections on a video frame."""
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = _box_to_ints(det["xyxy"], annotated.shape)
        if x2 <= x1 or y2 <= y1:
            continue

        measurement = measurements.get(det["track_id"])
        if measurement is None:
            color = (0, 180, 255)
            state = "pending"
            brightness = ""
        elif measurement.is_on:
            color = (40, 190, 70)
            state = "on"
            brightness = f" mean={measurement.mean_brightness:.0f}"
        else:
            color = (40, 40, 230)
            state = "off"
            brightness = f" mean={measurement.mean_brightness:.0f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"streetlight {det['track_id']} {det['confidence']:.2f} {state}{brightness}"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, 0.5, 1)
        label_y = max(y1, text_h + 8)
        label_x2 = min(annotated.shape[1] - 1, x1 + text_w + 8)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - 8),
            (label_x2, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - 4),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ================================================================== #
# Main pipeline                                                       #
# ================================================================== #

def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    # Dry-run: print config and exit (no file validation needed)
    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return

    if cfg.location_prior.prior_only:
        _run_prior_only(cfg)
        return

    if not cfg.video_path:
        print("ERROR: --video is required unless --prior-only is used", file=sys.stderr)
        sys.exit(2)
    if not cfg.detector.model_path:
        print("ERROR: --model is required unless --prior-only is used", file=sys.stderr)
        sys.exit(2)

    # Resolve output directory
    video_path = Path(cfg.video_path)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(__file__).resolve().parents[1] / "runs" / "audit" / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STREETLIGHT AUDIT PIPELINE")
    print("=" * 70)
    print(f"  Video:  {cfg.video_path}")
    print(f"  Model:  {cfg.detector.model_path}")
    print(f"  Output: {output_dir}")
    print("=" * 70)
    print()

    # ── Step 1: Load model ───────────────────────────────────────────
    print("[1/6] Loading YOLO model...")
    model = load_model(cfg.detector)
    try:
        resolved_classes = resolve_target_classes(model, cfg.detector)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    resolved_names = cfg.detector.resolved_class_names
    resolved_desc = ", ".join(
        f"{class_id}: {resolved_names.get(class_id, class_id)}"
        for class_id in resolved_classes
    )
    print(f"       Target class IDs: {resolved_desc}")

    # Save run config after resolving model-dependent class IDs.
    config_dict = cfg.to_dict()
    (output_dir / "run_config.json").write_text(
        json.dumps(config_dict, indent=2, default=str), encoding="utf-8"
    )

    # ── Step 2: Write tracker config ─────────────────────────────────
    print("[2/6] Configuring tracker...")
    tracker_yaml = write_tracker_config(cfg.tracker, output_dir)

    # ── Step 3: Stream video with tracking + filtering + measurement ─
    print("[3/6] Processing video frames...")
    video_meta = get_video_metadata(cfg.video_path)
    print(f"       Video: {video_meta['width']}x{video_meta['height']} "
          f"@ {video_meta['fps']:.1f} FPS, {video_meta['frame_count']} frames")

    gt_labels_dir = Path(cfg.evaluation.gt_labels_dir) if cfg.evaluation.gt_labels_dir else None
    if gt_labels_dir and not gt_labels_dir.exists():
        print(f"  WARNING: GT labels dir not found: {gt_labels_dir}", file=sys.stderr)
        gt_labels_dir = None

    # Accumulators
    all_measurements: list[LampMeasurement] = []
    all_filter_results: list[FilterResult] = []
    track_frame_counts: dict[str, int] = defaultdict(int)
    track_centers: dict[str, list[tuple[float, float]]] = defaultdict(list)
    track_histories: dict[str, list[list[float]]] = {}

    # For evaluation
    all_pred_boxes_raw: list[list[list[float]]] = []  # per-frame, before filtering
    all_pred_scores_raw: list[list[float]] = []
    all_pred_boxes_filtered: list[list[list[float]]] = []  # per-frame, after filtering
    all_pred_scores_filtered: list[list[float]] = []
    all_gt_boxes: list[list[list[float]]] = []

    total_raw_detections = 0
    total_kept_detections = 0
    total_model_detections = 0
    non_target_detections_suppressed = 0
    rejection_reasons: dict[str, int] = defaultdict(int)
    allowed_class_ids = set(resolved_classes)

    # Annotated video writer
    save_video = not args.no_save_video
    video_writer = None
    annotated_path = output_dir / "annotated.mp4"

    start_time = time.time()
    frame_count = 0

    for result in stream_video_with_tracking(
        model, cfg.video_path, cfg.detector, tracker_yaml, cfg.tracker
    ):
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_count / max(elapsed, 0.001)
            print(f"       Frame {frame_count}... ({fps_proc:.1f} frames/sec)")

        # Get original frame for brightness measurement
        frame = get_frame_from_result(result)
        if frame is None:
            all_pred_boxes_raw.append([])
            all_pred_scores_raw.append([])
            all_pred_boxes_filtered.append([])
            all_pred_scores_filtered.append([])
            if gt_labels_dir:
                gt_boxes = load_gt_boxes_yolo(
                    gt_labels_dir, frame_count, cfg.tracker.vid_stride,
                    video_meta["width"], video_meta["height"],
                )
                all_gt_boxes.append(gt_boxes)
            continue

        frame_height, frame_width = frame.shape[:2]

        # Extract raw model detections, then enforce the streetlight class gate.
        model_detections = extract_detections(result)
        total_model_detections += len(model_detections)
        detections, suppressed = keep_target_class_detections(
            model_detections, allowed_class_ids
        )
        non_target_detections_suppressed += suppressed
        total_raw_detections += len(detections)

        # Store raw predictions for evaluation
        all_pred_boxes_raw.append([d["xyxy"] for d in detections])
        all_pred_scores_raw.append([d["confidence"] for d in detections])

        # Load GT if available
        if gt_labels_dir:
            gt_boxes = load_gt_boxes_yolo(
                gt_labels_dir, frame_count, cfg.tracker.vid_stride,
                frame_width, frame_height,
            )
            all_gt_boxes.append(gt_boxes)

        # Multi-cue filtering
        if cfg.multicue.enabled:
            filter_results = filter_frame_detections(
                frame, detections, frame_count, cfg.multicue, track_histories
            )
            kept = [fr for fr in filter_results if fr.kept]
            rejected = [fr for fr in filter_results if not fr.kept]

            for fr in rejected:
                for reason in fr.reasons_rejected:
                    # Extract reason category (before '=')
                    category = reason.split("=")[0].strip()
                    rejection_reasons[category] += 1

            all_filter_results.extend(filter_results)
            total_kept_detections += len(kept)
        else:
            kept = []  # no filter results to track
            total_kept_detections += len(detections)

        # Store filtered predictions for evaluation
        if cfg.multicue.enabled:
            all_pred_boxes_filtered.append([fr.xyxy for fr in kept])
            all_pred_scores_filtered.append([fr.confidence for fr in kept])
        else:
            all_pred_boxes_filtered.append([d["xyxy"] for d in detections])
            all_pred_scores_filtered.append([d["confidence"] for d in detections])

        # Track centres for duplicate removal and frame counts for temporal filter
        active_dets: list[Detection]
        if cfg.multicue.enabled:
            det_by_track = {det["track_id"]: det for det in detections}
            active_dets = []
            for fr in kept:
                source = det_by_track.get(fr.track_id)
                active_dets.append({
                    "track_id": fr.track_id,
                    "xyxy": fr.xyxy,
                    "confidence": fr.confidence,
                    "class_id": source["class_id"] if source else None,
                    "class_label": source["class_label"] if source else "streetlight",
                })
        else:
            active_dets = detections

        frame_measurements: dict[str, LampMeasurement] = {}
        for det in active_dets:
            tid = det["track_id"]
            xyxy = det["xyxy"]
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            track_frame_counts[tid] += 1
            track_centers[tid].append((cx, cy))

            # Measure brightness
            m = measure_lamp(
                frame, tid, frame_count, xyxy, det["confidence"], cfg.measurement
            )
            all_measurements.append(m)
            frame_measurements[tid] = m

        # Save annotated video frame
        if save_video and result is not None:
            plotted = draw_streetlight_annotations(frame, active_dets, frame_measurements)
            if video_writer is None:
                h, w = plotted.shape[:2]
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer_fps = video_meta["fps"] if video_meta["fps"] > 0 else 30.0
                video_writer = cv2.VideoWriter(str(annotated_path), fourcc, writer_fps, (w, h))
            video_writer.write(plotted)

    if video_writer is not None:
        video_writer.release()

    elapsed = time.time() - start_time
    print(f"       Done — {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count / max(elapsed, 0.001):.1f} FPS)")
    print()

    # ── Step 4: Post-processing (temporal + duplicate filters) ───────
    print("[4/6] Temporal aggregation & post-processing...")

    # Temporal consistency filter: find tracks too short to trust
    short_tracks = temporal_consistency_filter(track_frame_counts, cfg.multicue)
    tracks_removed_temporal = len(short_tracks)

    # Remove measurements from short-lived tracks
    all_measurements = [m for m in all_measurements if m.track_id not in short_tracks]

    # Duplicate removal
    merge_map = duplicate_removal(track_centers, cfg.multicue)
    duplicates_merged = len(merge_map)

    # Aggregate
    lamps = aggregate_measurements(
        all_measurements, cfg.aggregation, cfg.measurement, merge_map
    )
    location_prior_report = _process_location_prior(
        lamps,
        cfg,
        output_dir,
        run_id=video_path.stem,
    )
    print(f"       {len(lamps)} unique lamps after aggregation")
    print(f"       {tracks_removed_temporal} tracks removed (too short)")
    print(f"       {duplicates_merged} duplicate tracks merged")
    if location_prior_report:
        prior_updates = location_prior_report["current_run_updates"]
        print(
            "       Location prior: "
            f"{prior_updates['matched_existing_candidates']} matched, "
            f"{prior_updates['new_candidates']} new candidates"
        )
    print()

    # ── Step 5: Evaluation (if GT available) ─────────────────────────
    eval_metrics: dict[str, Any] | None = None
    if gt_labels_dir and all_gt_boxes:
        print("[5/6] Computing evaluation metrics...")

        # Detection mAP (on filtered predictions)
        map_results = compute_map(
            all_pred_boxes_filtered,
            all_pred_scores_filtered,
            all_gt_boxes,
            cfg.evaluation.iou_thresholds,
        )

        # Aggregate frame-level detection metrics at IoU=0.5
        agg_det = DetectionMetrics(iou_threshold=0.5)
        for pred_b, pred_s, gt_b in zip(
            all_pred_boxes_filtered, all_pred_scores_filtered, all_gt_boxes
        ):
            frame_det = evaluate_frame_detections(pred_b, pred_s, gt_b, 0.5)
            agg_det.tp += frame_det.tp
            agg_det.fp += frame_det.fp
            agg_det.fn += frame_det.fn

        # Before-filtering metrics for comparison
        agg_before = DetectionMetrics(iou_threshold=0.5)
        for pred_b, pred_s, gt_b in zip(
            all_pred_boxes_raw, all_pred_scores_raw, all_gt_boxes
        ):
            frame_det = evaluate_frame_detections(pred_b, pred_s, gt_b, 0.5)
            agg_before.tp += frame_det.tp
            agg_before.fp += frame_det.fp
            agg_before.fn += frame_det.fn

        total_gt_instances = sum(len(gt) for gt in all_gt_boxes)
        raw_prediction_instances = sum(len(pred) for pred in all_pred_boxes_raw)
        filtered_prediction_instances = sum(len(pred) for pred in all_pred_boxes_filtered)
        count_error_pct = (
            abs(filtered_prediction_instances - total_gt_instances)
            / total_gt_instances
            * 100.0
            if total_gt_instances > 0
            else 0.0
        )

        eval_metrics = {
            "evaluation_context": {
                "frames_evaluated": len(all_gt_boxes),
                "frames_with_ground_truth": sum(1 for gt in all_gt_boxes if gt),
                "ground_truth_instances": total_gt_instances,
                "raw_target_prediction_instances": raw_prediction_instances,
                "filtered_target_prediction_instances": filtered_prediction_instances,
                "unique_lamps_after_tracking": len(lamps),
                "ground_truth_labels_dir": str(gt_labels_dir),
                "note": (
                    "Detection metrics are frame-level bounding-box metrics. "
                    "Unique physical lamp accuracy requires tracked ground truth IDs."
                ),
            },
            "detection": map_results,
            "frame_detection": agg_det.to_dict(),
            "before_after_filtering": {
                "before": agg_before.to_dict(),
                "after": agg_det.to_dict(),
            },
            "frame_instance_counts": {
                "ground_truth_instances": total_gt_instances,
                "filtered_prediction_instances": filtered_prediction_instances,
                "count_error_pct": round(count_error_pct, 2),
                "matched_instances_tp_at_iou_0_5": agg_det.tp,
                "extra_instances_fp_at_iou_0_5": agg_det.fp,
                "missed_instances_fn_at_iou_0_5": agg_det.fn,
            },
        }

        # Status classification (if GT status file provided)
        if cfg.evaluation.gt_status_file and Path(cfg.evaluation.gt_status_file).exists():
            import csv as csv_mod
            gt_statuses: dict[str, str] = {}
            with open(cfg.evaluation.gt_status_file, "r") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    gt_statuses[row["lamp_id"]] = row["status"]

            status_metrics = StatusClassificationMetrics()
            for lamp in lamps:
                gt_status = gt_statuses.get(lamp.track_id)
                if gt_status is None:
                    continue
                pred_working = lamp.status == "working"
                gt_working = gt_status.lower() in ("working", "on", "1", "true")
                if pred_working and gt_working:
                    status_metrics.tp += 1
                elif pred_working and not gt_working:
                    status_metrics.fp += 1
                elif not pred_working and not gt_working:
                    status_metrics.tn += 1
                else:
                    status_metrics.fn += 1

            eval_metrics["status_classification"] = status_metrics.to_dict()

        print(f"       mAP@0.5: {map_results['AP@0.5']}")
        print(f"       mAP@0.5:0.95: {map_results['mAP@0.5:0.95']}")
        print(f"       Precision: {agg_det.precision:.4f}")
        print(f"       Recall: {agg_det.recall:.4f}")
        print(f"       F1: {agg_det.f1:.4f}")
        print(f"       TP/FP/FN @ IoU=0.5: {agg_det.tp}/{agg_det.fp}/{agg_det.fn}")
        print()
    else:
        print("[5/6] Skipping evaluation (no ground-truth labels provided)")
        print()

    # ── Step 6: Generate report ──────────────────────────────────────
    print("[6/6] Generating audit report...")

    filter_stats = {
        "total_model_detections": total_model_detections,
        "total_raw_detections": total_raw_detections,
        "total_kept_detections": total_kept_detections,
        "non_target_detections_suppressed": non_target_detections_suppressed,
        "rejection_rate_pct": round(
            (1.0 - total_kept_detections / max(total_raw_detections, 1)) * 100, 2
        ),
        "rejection_reasons": dict(rejection_reasons),
        "temporal_filtering": {
            "tracks_removed": tracks_removed_temporal,
            "duplicates_merged": duplicates_merged,
        },
    }
    if location_prior_report:
        filter_stats["location_prior"] = {
            "report_path": location_prior_report.get("report_path"),
            "query_claim": location_prior_report.get("query", {}).get("claim"),
            "known_lamp_candidates": location_prior_report.get("prior_summary", {}).get("known_lamp_candidates"),
            "location_evidence_count": location_prior_report.get("current_run_updates", {}).get("location_evidence_count"),
            "matched_existing_candidates": location_prior_report.get("current_run_updates", {}).get("matched_existing_candidates"),
            "new_candidates": location_prior_report.get("current_run_updates", {}).get("new_candidates"),
        }

    json_path = write_json_report(output_dir, lamps, config_dict, video_meta, filter_stats, eval_metrics, location_prior_report)
    csv_path = write_csv_report(output_dir, lamps)
    md_path = write_markdown_report(output_dir, lamps, config_dict, video_meta, filter_stats, eval_metrics, location_prior_report)

    # Also save per-lamp detailed JSON
    per_lamp_path = output_dir / "per_lamp_data.json"
    per_lamp_path.write_text(
        json.dumps([l.to_dict() for l in lamps], indent=2), encoding="utf-8"
    )

    # Save filter statistics
    filter_stats_path = output_dir / "filter_stats.json"
    filter_stats_path.write_text(
        json.dumps(filter_stats, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("  AUDIT COMPLETE")
    print("=" * 70)
    print()
    total = len(lamps)
    working = sum(1 for l in lamps if l.status == "working")
    off = sum(1 for l in lamps if l.status == "off")
    flickering = sum(1 for l in lamps if l.status == "flickering")
    print(f"  Total lamps detected:  {total}")
    print(f"  ✅ Working (on):       {working}")
    print(f"  ❌ Off / faulty:       {off}")
    print(f"  ⚡ Flickering:         {flickering}")
    if total > 0:
        print(f"  Working rate:          {working / total * 100:.1f}%")
    print()
    print("  Output files:")
    print(f"    📄 {json_path}")
    print(f"    📊 {csv_path}")
    print(f"    📝 {md_path}")
    print(f"    🔍 {per_lamp_path}")
    print(f"    🔧 {filter_stats_path}")
    if save_video and annotated_path.exists():
        print(f"    🎥 {annotated_path}")
    print()


if __name__ == "__main__":
    main()
