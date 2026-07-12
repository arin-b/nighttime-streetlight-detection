"""Multi-cue detection filtering (PDF §4).

The audit pipeline now uses weighted cue scoring that mirrors the video-runner
pipeline, while still keeping the temporal consistency and duplicate-removal
steps used later in the audit flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from evaluation.eval_pres.audit_config import MultiCueConfig
from rbccps_od.config.schemas import CueWeights
from rbccps_od.domain.cues import CueScore
from rbccps_od.pipeline.aggregator import weighted_aggregate
from rbccps_od.pipeline.thresholding import threshold_score

if TYPE_CHECKING:
    from evaluation.eval_pres.run_audit import Detection


# ------------------------------------------------------------------ #
# Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class FilterResult:
    """Result of running all multi-cue filters on one detection."""
    track_id: str
    frame_index: int
    xyxy: list[float]
    confidence: float
    kept: bool = True
    aggregate_score: float = 0.0
    reasons_rejected: list[str] = field(default_factory=list)
    cue_scores: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Weighted cue scoring logic (mirrors the video-runner pipeline)      #
# ------------------------------------------------------------------ #

def _trajectory_cue(history_len: int) -> CueScore:
    score = min(1.0, history_len / 4.0)
    return CueScore(name="trajectory", value=score, weight=1.0, metadata={"history_len": history_len})


def _size_progression_cue(history: list[list[float]]) -> CueScore:
    areas = []
    for xyxy in history:
        x1, y1, x2, y2 = xyxy
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        areas.append(width * height)

    if len(areas) < 2:
        score = 1.0 if (areas and areas[0] > 0) else 0.0
    else:
        ratios = [curr / prev if prev > 0 else 1.0 for prev, curr in zip(areas[:-1], areas[1:])]
        ema = ratios[0]
        alpha = 0.3
        for r in ratios[1:]:
            ema = alpha * r + (1 - alpha) * ema
        score = 1.0 if ema >= 0.95 else max(0.0, ema / 0.95)
        
    return CueScore(name="size_progression", value=score, weight=1.0, metadata={"areas": areas})


def _light_characteristics_cue(confidence: float) -> CueScore:
    score = max(0.0, min(1.0, confidence))
    return CueScore(name="light_characteristics", value=score, weight=1.0, metadata={"confidence": confidence})


def _position_prior_cue(xyxy: list[float], frame_height: int | None) -> CueScore:
    _, y1, _, y2 = xyxy
    center_y = (y1 + y2) / 2.0
    if frame_height and frame_height > 0:
        normalised = center_y / frame_height
        score = 1.0 - abs(normalised - 0.4)
    else:
        score = 0.5
    score = max(0.0, min(1.0, score))
    return CueScore(name="position_prior", value=score, weight=1.0, metadata={"center_y": center_y})


def _score_track(
    xyxy: list[float],
    confidence: float,
    track_histories: dict[str, list[list[float]]],
    track_id: str,
    frame_height: int | None,
    weights: CueWeights | None = None,
) -> tuple[float, dict[str, float]]:
    history = track_histories.setdefault(track_id, [])
    history.append(list(xyxy))
    history_len = len(history)

    cues = [
        _trajectory_cue(history_len),
        _size_progression_cue(history),
        _light_characteristics_cue(confidence),
        _position_prior_cue(xyxy, frame_height),
    ]
    if weights is None:
        weights = CueWeights()
    for cue in cues:
        cue.weight = getattr(weights, cue.name)

    aggregate_score = weighted_aggregate(cues)
    cue_values = {cue.name: float(cue.value) for cue in cues}
    return aggregate_score, cue_values


# ------------------------------------------------------------------ #
# Individual filters                                                  #
# ------------------------------------------------------------------ #

def _aspect_ratio_ok(xyxy: list[float], cfg: MultiCueConfig) -> tuple[bool, float]:
    """Check that box width/height ratio is within plausible range."""
    x1, y1, x2, y2 = xyxy
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    ratio = w / h
    ok = cfg.aspect_ratio_min <= ratio <= cfg.aspect_ratio_max
    return ok, round(ratio, 4)


def _spatial_location_ok(
    xyxy: list[float], frame_height: int, cfg: MultiCueConfig
) -> tuple[bool, float]:
    """Reject detections whose centre is in the bottom portion of the frame
    (below the assumed road horizon)."""
    _, y1, _, y2 = xyxy
    center_y = (y1 + y2) / 2.0
    normalised = center_y / max(frame_height, 1)
    ok = normalised <= cfg.spatial_upper_fraction
    return ok, round(normalised, 4)


def _brightness_cue_ok(
    frame: np.ndarray,
    xyxy: list[float],
    cfg: MultiCueConfig,
) -> tuple[bool, float]:
    """Check that the ROI has at least *some* bright pixels (lit lamp)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(xyxy[0])),
        max(0, int(xyxy[1])),
        min(w, int(xyxy[2])),
        min(h, int(xyxy[3])),
    )
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    max_val = float(np.max(gray))
    ok = max_val >= cfg.min_brightness_for_detection
    return ok, round(max_val, 2)


# ------------------------------------------------------------------ #
# Track-level temporal filters (applied after all frames are seen)   #
# ------------------------------------------------------------------ #

def temporal_consistency_filter(
    track_frame_counts: dict[str, int],
    cfg: MultiCueConfig,
) -> set[str]:
    """Return the set of track_ids that are too short-lived to trust."""
    return {
        tid
        for tid, count in track_frame_counts.items()
        if count < cfg.min_track_frames_for_confirmation
    }


def duplicate_removal(
    track_centers: dict[str, list[tuple[float, float]]],
    cfg: MultiCueConfig,
) -> dict[str, str]:
    """Find pairs of tracks whose centres stay within *duplicate_center_distance_px*
    across all shared frames and merge the shorter one into the longer one.

    Returns a mapping ``{absorbed_track_id: canonical_track_id}``.
    """
    merge_map: dict[str, str] = {}
    track_ids = sorted(track_centers.keys())

    for i, tid_a in enumerate(track_ids):
        if tid_a in merge_map:
            continue
        for tid_b in track_ids[i + 1:]:
            if tid_b in merge_map:
                continue
            centers_a = track_centers[tid_a]
            centers_b = track_centers[tid_b]
            # Only compare if they overlap in time
            min_len = min(len(centers_a), len(centers_b))
            if min_len == 0:
                continue
            dists = [
                math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                for ca, cb in zip(centers_a[-min_len:], centers_b[-min_len:])
            ]
            if max(dists) < cfg.duplicate_center_distance_px:
                # Absorb the shorter track
                if len(centers_a) >= len(centers_b):
                    merge_map[tid_b] = tid_a
                else:
                    merge_map[tid_a] = tid_b

    return merge_map


# ------------------------------------------------------------------ #
# Per-frame filter runner                                             #
# ------------------------------------------------------------------ #

def filter_frame_detections(
    frame: np.ndarray,
    detections: list[Detection],
    frame_index: int,
    cfg: MultiCueConfig,
    track_histories: dict[str, list[list[float]]] | None = None,
) -> list[FilterResult]:
    """Apply per-frame multi-cue filters to a list of detections.

    Each detection dict must have keys: ``track_id``, ``xyxy``, ``confidence``.

    Returns a list of ``FilterResult`` objects with *kept* set to True/False.
    """
    frame_height = frame.shape[0]
    track_histories = track_histories or {}
    results: list[FilterResult] = []

    for det in detections:
        tid = det["track_id"]
        xyxy = det["xyxy"]
        conf = det["confidence"]
        fr = FilterResult(
            track_id=tid,
            frame_index=frame_index,
            xyxy=xyxy,
            confidence=conf,
        )

        aggregate_score, cue_values = _score_track(
            xyxy,
            conf,
            track_histories,
            tid,
            frame_height,
        )
        fr.aggregate_score = aggregate_score
        fr.cue_scores.update(cue_values)
        fr.kept = threshold_score(aggregate_score, cfg.aggregation_threshold)
        if not fr.kept:
            fr.reasons_rejected.append(
                f"aggregate_score={aggregate_score:.3f} < {cfg.aggregation_threshold:.2f}"
            )

        results.append(fr)

    return results
