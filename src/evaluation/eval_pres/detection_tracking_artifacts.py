from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .artifact_utils import read_json, relative_or_absolute, safe_float, write_csv, write_html_table, write_json
from .io import load_ground_truth, load_manifest_predictions
from .metrics import bbox_iou, match_predictions
from .schemas import BoxRecord


def build_detection_tracking_artifacts(
    manifest_path: str | Path,
    out_dir: str | Path,
    frame_root: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    render_overlays: bool = True,
    video_fps: float = 3.0,
) -> dict[str, str]:
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_root_path = Path(frame_root) if frame_root is not None else manifest_path.parent
    tracks = list(manifest.get("tracks", []))
    frames = list(manifest.get("frames", []))
    predictions = load_manifest_predictions(manifest_path)
    ground_truth = load_ground_truth(ground_truth_path)

    detection_rows = _detection_rows(tracks, predictions, ground_truth)
    frame_rows = _frame_detection_rows(frames, tracks)
    track_frame_rows = _track_frame_rows(tracks)
    track_rows = _track_summary_rows(tracks)
    fragmentation_rows = _fragmentation_rows(tracks)
    duplicate_rows = _duplicate_candidate_rows(tracks)

    artifacts = {
        "detection_summary_table_csv": str(write_csv(out / "detection_summary_table.csv", detection_rows)),
        "detection_summary_table_html": str(write_html_table(out / "detection_summary_table.html", "Detection Summary", detection_rows)),
        "frame_detection_counts_csv": str(write_csv(out / "frame_detection_counts.csv", frame_rows)),
        "track_summary_table_csv": str(write_csv(out / "track_summary_table.csv", track_rows)),
        "track_summary_csv": str(write_csv(out / "track_summary.csv", track_rows)),
        "track_frame_table_csv": str(write_csv(out / "track_frame_table.csv", track_frame_rows)),
        "tracking_events_csv": str(write_csv(out / "tracking_events.csv", track_frame_rows)),
        "track_fragmentation_table_csv": str(write_csv(out / "track_fragmentation_table.csv", fragmentation_rows)),
        "duplicate_track_candidates_csv": str(write_csv(out / "duplicate_track_candidates.csv", duplicate_rows)),
    }
    if ground_truth:
        gt_tables = _ground_truth_tables(predictions, ground_truth)
        artifacts.update({
            "detection_gt_match_table_csv": str(write_csv(out / "detection_gt_match_table.csv", gt_tables["matches"])),
            "false_positive_candidates_csv": str(write_csv(out / "false_positive_candidates.csv", gt_tables["false_positives"])),
            "missed_ground_truth_lamps_csv": str(write_csv(out / "missed_ground_truth_lamps.csv", gt_tables["missed"])),
            "track_gt_match_table_csv": str(write_csv(out / "track_gt_match_table.csv", _track_gt_rows(predictions, ground_truth))),
        })

    plot_dir = out / "detection_tracking_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    artifacts.update(_write_detection_plots(plot_dir, tracks, frame_rows, track_rows, duplicate_rows, predictions, ground_truth))

    if render_overlays:
        overlay_artifacts = render_detection_tracking_overlays(
            manifest,
            frame_root_path,
            out,
            video_fps=video_fps,
        )
        artifacts.update(overlay_artifacts)

    write_json(out / "detection_tracking_artifacts_manifest.json", artifacts)
    return artifacts


def render_detection_tracking_overlays(
    manifest: dict[str, Any],
    frame_root: Path,
    out: Path,
    video_fps: float = 3.0,
) -> dict[str, str]:
    detection_dir = out / "detections_overlay_frames"
    tracking_dir = out / "tracking_overlay_frames"
    detection_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    frames = list(manifest.get("frames", []))
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest.get("tracks", []):
        by_frame[int(row.get("frame_id", 0))].append(row)
    detection_frames: list[Path] = []
    tracking_frames: list[Path] = []
    font = _font(16)
    small = _font(12)
    for frame in frames:
        frame_id = int(frame.get("frame_id", 0))
        image_path = relative_or_absolute(Path(str(frame.get("image_uri"))), frame_root)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        det_image = image.copy()
        trk_image = image.copy()
        det_draw = ImageDraw.Draw(det_image)
        trk_draw = ImageDraw.Draw(trk_image)
        _draw_banner(det_draw, det_image.size, f"Detections | frame {frame_id}", font)
        _draw_banner(trk_draw, trk_image.size, f"Tracking | frame {frame_id}", font)
        for item in by_frame.get(frame_id, []):
            bbox = [float(v) for v in item.get("bbox_xyxy", [0, 0, 0, 0])]
            score = safe_float(item.get("detector_score"))
            track_id = str(item.get("track_id", ""))
            color = _track_color(track_id)
            det_draw.rectangle(bbox, outline=color, width=3)
            trk_draw.rectangle(bbox, outline=color, width=3)
            det_draw.text((bbox[0] + 3, max(30, bbox[1] - 16)), f"det {score:.2f}", fill=color, font=small)
            trk_draw.text((bbox[0] + 3, max(30, bbox[1] - 16)), f"{track_id} {score:.2f}", fill=color, font=small)
        det_path = detection_dir / f"detections_{frame_id:06d}.jpg"
        trk_path = tracking_dir / f"tracking_{frame_id:06d}.jpg"
        det_image.save(det_path, quality=90)
        trk_image.save(trk_path, quality=90)
        detection_frames.append(det_path)
        tracking_frames.append(trk_path)
    artifacts: dict[str, str] = {}
    if detection_frames:
        artifacts["detections_overlay_video"] = str(_write_mp4(detection_frames, out / "detections_overlay_video.mp4", video_fps))
        artifacts["tracking_overlay_video"] = str(_write_mp4(tracking_frames, out / "tracking_overlay_video.mp4", video_fps))
        artifacts["tracking_contact_sheet"] = str(_contact_sheet(tracking_frames, out / "tracking_contact_sheet.jpg"))
    _write_track_cards(manifest, frame_root, out / "track_cards")
    artifacts["detections_overlay_frames"] = str(detection_dir)
    artifacts["tracking_overlay_frames"] = str(tracking_dir)
    artifacts["track_cards"] = str(out / "track_cards")
    return artifacts


def _detection_rows(tracks: list[dict[str, Any]], predictions: tuple[BoxRecord, ...], gt: tuple[BoxRecord, ...]) -> list[dict[str, Any]]:
    matches = match_predictions(predictions, gt, 0.5) if gt else []
    match_by_pred = {match.pred_index: match for match in matches}
    rows = []
    for index, item in enumerate(tracks):
        bbox = item.get("bbox_xyxy", [None, None, None, None])
        match = match_by_pred.get(index)
        rows.append({
            "detection_index": index,
            "frame_id": item.get("frame_id"),
            "track_id": item.get("track_id"),
            "class_name": item.get("class_name"),
            "detector_score": item.get("detector_score"),
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
            "bbox_area": _bbox_area(bbox),
            "gt_match_index": match.gt_index if match else None,
            "gt_iou": round(match.iou, 4) if match else None,
            "match_status": "tp" if match and match.gt_index is not None else ("fp" if gt else "unlabeled"),
        })
    return rows


def _frame_detection_rows(frames: list[dict[str, Any]], tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in tracks:
        by_frame[int(item.get("frame_id", 0))].append(item)
    frame_ids = sorted({int(frame.get("frame_id", 0)) for frame in frames} | set(by_frame))
    rows = []
    for frame_id in frame_ids:
        items = by_frame.get(frame_id, [])
        scores = [safe_float(row.get("detector_score")) for row in items]
        rows.append({
            "frame_id": frame_id,
            "detection_count": len(items),
            "mean_confidence": round(float(np.mean(scores)), 4) if scores else 0.0,
            "max_confidence": round(max(scores), 4) if scores else 0.0,
        })
    return rows


def _track_frame_rows(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in tracks:
        bbox = item.get("bbox_xyxy", [None, None, None, None])
        rows.append({
            "track_id": item.get("track_id"),
            "frame_id": item.get("frame_id"),
            "timestamp_ns": item.get("timestamp_ns"),
            "detector_score": item.get("detector_score"),
            "track_confidence": item.get("track_confidence"),
            "bbox_xyxy": bbox,
            "bbox_area": _bbox_area(bbox),
        })
    return rows


def _track_summary_rows(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by_track(tracks)
    rows = []
    for track_id, items in sorted(grouped.items()):
        frame_ids = sorted(int(item.get("frame_id", 0)) for item in items)
        scores = [safe_float(item.get("detector_score")) for item in items]
        areas = [_bbox_area(item.get("bbox_xyxy", [0, 0, 0, 0])) for item in items]
        rows.append({
            "track_id": track_id,
            "first_frame": frame_ids[0],
            "last_frame": frame_ids[-1],
            "track_length_frames": len(items),
            "span_frames": frame_ids[-1] - frame_ids[0] + 1,
            "continuity": round(len(items) / max(1, frame_ids[-1] - frame_ids[0] + 1), 4),
            "mean_confidence": round(float(np.mean(scores)), 4),
            "max_confidence": round(max(scores), 4),
            "mean_bbox_area": round(float(np.mean(areas)), 2),
        })
    return rows


def _fragmentation_rows(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for track_id, items in sorted(_group_by_track(tracks).items()):
        frame_ids = sorted(int(item.get("frame_id", 0)) for item in items)
        gaps = [b - a - 1 for a, b in zip(frame_ids, frame_ids[1:]) if b - a > 1]
        rows.append({
            "track_id": track_id,
            "gap_count": len(gaps),
            "missing_frame_count": sum(gaps),
            "max_gap": max(gaps) if gaps else 0,
            "fragmentation_flag": int(bool(gaps)),
        })
    return rows


def _duplicate_candidate_rows(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by_track(tracks)
    summaries = {track_id: _track_descriptor(items) for track_id, items in grouped.items()}
    rows = []
    ids = sorted(summaries)
    for idx, left in enumerate(ids):
        for right in ids[idx + 1 :]:
            score = _duplicate_score(summaries[left], summaries[right])
            if score >= 0.35:
                rows.append({
                    "track_id_a": left,
                    "track_id_b": right,
                    "duplicate_likelihood": round(score, 4),
                    "temporal_overlap": round(_temporal_overlap(summaries[left], summaries[right]), 4),
                    "center_distance_px": round(_center_distance(summaries[left]["center"], summaries[right]["center"]), 2),
                })
    rows.sort(key=lambda row: row["duplicate_likelihood"], reverse=True)
    return rows


def _ground_truth_tables(predictions: tuple[BoxRecord, ...], gt: tuple[BoxRecord, ...]) -> dict[str, list[dict[str, Any]]]:
    matches = match_predictions(predictions, gt, 0.5)
    matched_gt = {match.gt_index for match in matches if match.gt_index is not None}
    return {
        "matches": [
            {
                "pred_index": match.pred_index,
                "gt_index": match.gt_index,
                "frame_id": predictions[match.pred_index].frame_id,
                "track_id": predictions[match.pred_index].track_id,
                "iou": round(match.iou, 4),
                "status": "tp" if match.gt_index is not None else "fp",
            }
            for match in matches
        ],
        "false_positives": [
            {
                "pred_index": match.pred_index,
                "frame_id": predictions[match.pred_index].frame_id,
                "track_id": predictions[match.pred_index].track_id,
                "score": predictions[match.pred_index].score,
                "best_iou": round(match.iou, 4),
            }
            for match in matches
            if match.gt_index is None
        ],
        "missed": [
            {
                "gt_index": index,
                "frame_id": item.frame_id,
                "physical_lamp_id": item.physical_lamp_id,
                "inventory_id": item.inventory_id,
                "bbox_xyxy": item.bbox_xyxy,
            }
            for index, item in enumerate(gt)
            if index not in matched_gt
        ],
    }


def _track_gt_rows(predictions: tuple[BoxRecord, ...], gt: tuple[BoxRecord, ...]) -> list[dict[str, Any]]:
    matches = match_predictions(predictions, gt, 0.5)
    grouped: dict[str, list[str]] = defaultdict(list)
    for match in matches:
        if match.gt_index is None:
            continue
        pred = predictions[match.pred_index]
        grouped[pred.track_id or f"pred_{match.pred_index}"].append(gt[match.gt_index].physical_lamp_id or str(match.gt_index))
    rows = []
    for track_id, ids in sorted(grouped.items()):
        rows.append({
            "track_id": track_id,
            "matched_frames": len(ids),
            "dominant_gt_id": max(set(ids), key=ids.count),
            "unique_gt_ids": len(set(ids)),
            "identity_switch_count": sum(1 for a, b in zip(ids, ids[1:]) if a != b),
        })
    return rows


def _write_detection_plots(
    out: Path,
    tracks: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    track_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    predictions: tuple[BoxRecord, ...],
    gt: tuple[BoxRecord, ...],
) -> dict[str, str]:
    artifacts = {
        "detections_per_frame_plot": str(_plot_line(out / "detections_per_frame.png", [r["frame_id"] for r in frame_rows], [r["detection_count"] for r in frame_rows], "Detections Per Frame", "Frame", "Detections")),
        "detector_confidence_distribution_plot": str(_plot_hist(out / "detector_confidence_distribution.png", [safe_float(t.get("detector_score")) for t in tracks], "Detector Confidence Distribution", "Confidence")),
        "track_length_distribution_plot": str(_plot_hist(out / "track_length_distribution.png", [r["track_length_frames"] for r in track_rows], "Track Length Distribution", "Frames")),
        "track_confidence_vs_length_plot": str(_plot_scatter(out / "track_confidence_vs_length.png", [r["track_length_frames"] for r in track_rows], [r["mean_confidence"] for r in track_rows], "Track Length vs Confidence", "Track length", "Mean confidence")),
        "track_timeline_plot": str(_plot_track_timeline(out / "track_timeline.png", track_rows)),
        "bbox_area_distribution_plot": str(_plot_hist(out / "bbox_area_distribution.png", [_bbox_area(t.get("bbox_xyxy", [0, 0, 0, 0])) for t in tracks], "BBox Area Distribution", "Area px^2")),
        "detection_spatial_heatmap_plot": str(_plot_spatial(out / "detection_spatial_heatmap.png", tracks)),
    }
    if duplicate_rows:
        artifacts["duplicate_track_heatmap_plot"] = str(_plot_duplicate_heatmap(out / "duplicate_track_heatmap.png", duplicate_rows))
    if gt:
        thresholds = [round(v, 2) for v in np.arange(0.5, 0.951, 0.05)]
        aps = []
        from .metrics import _average_precision  # local use for presentation plot
        for threshold in thresholds:
            aps.append(_average_precision(predictions, gt, threshold) or 0.0)
        artifacts["ap_by_iou_threshold_plot"] = str(_plot_line(out / "ap_by_iou_threshold.png", thresholds, aps, "AP By IoU Threshold", "IoU threshold", "AP"))
    return artifacts


def _plot_line(path: Path, x: list[Any], y: list[Any], title: str, xlabel: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o", linewidth=1.8, color="#3182bd")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_hist(path: Path, values: list[float], title: str, xlabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values or [0], bins=min(20, max(5, len(values) // 2 or 5)), color="#74a9cf", edgecolor="#243b53")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_scatter(path: Path, x: list[float], y: list[float], title: str, xlabel: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, color="#2ca25f", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_track_timeline(path: Path, rows: list[dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.24 * len(rows))))
    for idx, row in enumerate(rows):
        ax.plot([row["first_frame"], row["last_frame"]], [idx, idx], linewidth=4)
    ax.set_yticks(range(len(rows)), [str(row["track_id"]) for row in rows], fontsize=7)
    ax.set_xlabel("Frame")
    ax.set_title("Track Timeline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_spatial(path: Path, tracks: list[dict[str, Any]]) -> Path:
    centers = []
    for item in tracks:
        bbox = item.get("bbox_xyxy", [0, 0, 0, 0])
        centers.append(((float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2))
    fig, ax = plt.subplots(figsize=(7, 4))
    if centers:
        x, y = zip(*centers)
        ax.hist2d(x, y, bins=25, cmap="Blues")
        ax.invert_yaxis()
    ax.set_title("Detection Spatial Heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_duplicate_heatmap(path: Path, rows: list[dict[str, Any]]) -> Path:
    ids = sorted({row["track_id_a"] for row in rows} | {row["track_id_b"] for row in rows})
    index = {track_id: idx for idx, track_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    for row in rows:
        a = index[row["track_id_a"]]
        b = index[row["track_id_b"]]
        matrix[a, b] = matrix[b, a] = row["duplicate_likelihood"]
    fig, ax = plt.subplots(figsize=(max(5, 0.3 * len(ids)), max(4, 0.3 * len(ids))))
    ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1)
    ax.set_title("Duplicate Track Candidate Heatmap")
    ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ids)), ids, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_mp4(frames: list[Path], path: Path, fps: float) -> Path:
    try:
        import cv2

        first = cv2.imread(str(frames[0]))
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            for frame in frames:
                image = cv2.imread(str(frame))
                if image is not None:
                    writer.write(image)
        finally:
            writer.release()
    except ImportError:
        _write_mp4_with_ffmpeg(frames, path, fps)
    return path


def _write_mp4_with_ffmpeg(frames: list[Path], path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("MP4 rendering requires either opencv-python or ffmpeg on PATH")
    staging = path.parent / f"_{path.stem}_frames"
    staging.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        shutil.copy2(frame, staging / f"frame_{index:06d}.jpg")
    pattern = staging / "frame_%06d.jpg"
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(pattern), "-pix_fmt", "yuv420p", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(staging, ignore_errors=True)


def _contact_sheet(frames: list[Path], path: Path, columns: int = 4, rows: int = 4) -> Path:
    selected = frames[:: max(1, len(frames) // (columns * rows))][: columns * rows]
    thumbs = []
    for frame in selected:
        image = Image.open(frame).convert("RGB")
        image.thumbnail((320, 180), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (320, 180), (20, 20, 20))
        canvas.paste(image, ((320 - image.width) // 2, (180 - image.height) // 2))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (columns * 320, rows * 180), (35, 35, 35))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % columns) * 320, (idx // columns) * 180))
    sheet.save(path)
    return path


def _write_track_cards(manifest: dict[str, Any], frame_root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frames_by_id = {int(frame["frame_id"]): frame for frame in manifest.get("frames", [])}
    for track_id, items in _group_by_track(manifest.get("tracks", [])).items():
        first = sorted(items, key=lambda row: int(row.get("frame_id", 0)))[0]
        frame = frames_by_id.get(int(first.get("frame_id", 0)))
        if not frame:
            continue
        image_path = relative_or_absolute(Path(str(frame.get("image_uri"))), frame_root)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        bbox = [int(float(v)) for v in first.get("bbox_xyxy", [0, 0, 0, 0])]
        crop = image.crop((max(0, bbox[0] - 20), max(0, bbox[1] - 20), min(image.width, bbox[2] + 20), min(image.height, bbox[3] + 20)))
        crop.thumbnail((360, 240), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (420, 340), (245, 247, 250))
        card.paste(crop, ((420 - crop.width) // 2, 16))
        draw = ImageDraw.Draw(card)
        draw.text((18, 270), f"Track: {track_id}", fill=(20, 33, 61), font=_font(16))
        draw.text((18, 294), f"Frames: {len(items)} | Mean conf: {np.mean([safe_float(i.get('detector_score')) for i in items]):.2f}", fill=(20, 33, 61), font=_font(13))
        png = out / f"{track_id}.png"
        card.save(png)
        html = out / f"{track_id}.html"
        html.write_text(f"<html><body><h1>{track_id}</h1><img src='{png.name}'><pre>{items}</pre></body></html>", encoding="utf-8")


def _group_by_track(tracks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in tracks:
        grouped[str(item.get("track_id", ""))].append(item)
    return dict(grouped)


def _track_descriptor(items: list[dict[str, Any]]) -> dict[str, Any]:
    frame_ids = sorted(int(item.get("frame_id", 0)) for item in items)
    centers = []
    for item in items:
        bbox = item.get("bbox_xyxy", [0, 0, 0, 0])
        centers.append(((float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2))
    return {"first": frame_ids[0], "last": frame_ids[-1], "center": tuple(np.mean(centers, axis=0))}


def _duplicate_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    temporal = _temporal_overlap(a, b)
    distance = _center_distance(a["center"], b["center"])
    spatial = max(0.0, 1.0 - distance / 160.0)
    return max(0.0, min(1.0, 0.55 * spatial + 0.45 * temporal))


def _temporal_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = max(0, min(a["last"], b["last"]) - max(a["first"], b["first"]) + 1)
    span = max(a["last"], b["last"]) - min(a["first"], b["first"]) + 1
    return overlap / max(1, span)


def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox_area(bbox: Any) -> float:
    try:
        return round(max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1])), 2)
    except Exception:
        return 0.0


def _draw_banner(draw: ImageDraw.ImageDraw, size: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw.rectangle((0, 0, size[0], 28), fill=(0, 0, 0))
    draw.text((10, 6), text, fill=(255, 255, 255), font=font)


def _track_color(track_id: str) -> tuple[int, int, int]:
    seed = sum(ord(ch) for ch in track_id)
    return ((seed * 37) % 205 + 50, (seed * 53) % 205 + 50, (seed * 97) % 205 + 50)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RBCCPS detection/tracking presentation artifacts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-root")
    parser.add_argument("--ground-truth")
    parser.add_argument("--video-fps", type=float, default=3.0)
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_detection_tracking_artifacts(
        manifest_path=args.manifest,
        out_dir=args.out,
        frame_root=args.frame_root,
        ground_truth_path=args.ground_truth,
        render_overlays=not args.no_overlays,
        video_fps=args.video_fps,
    )
    print(write_json(Path(args.out) / "run_output.json", artifacts).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
