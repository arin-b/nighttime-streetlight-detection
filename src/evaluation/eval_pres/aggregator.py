"""Temporal aggregation (PDF §6).

Groups per-frame measurements by track ID, applies persistence checks,
computes per-lamp statistics, and decides final working status.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from evaluation.eval_pres.audit_config import AggregationConfig, MeasurementConfig
from evaluation.eval_pres.measurement import LampMeasurement


@dataclass
class AggregatedLamp:
    """Final per-lamp summary after temporal aggregation."""

    track_id: str
    status: str  # "working", "off", "flickering"
    # Brightness statistics
    avg_brightness: float = 0.0
    median_brightness: float = 0.0
    brightness_std: float = 0.0
    min_brightness: float = 0.0
    max_brightness: float = 0.0
    peak_brightness_mean: float = 0.0
    total_flux_mean: float = 0.0
    # Confidence
    confidence: float = 0.0
    # Representative bounding box (from highest-confidence frame)
    representative_bbox: list[float] = field(default_factory=list)
    # Frame span
    frame_count: int = 0
    first_frame: int = 0
    last_frame: int = 0
    frames_on: int = 0
    frames_off: int = 0
    # Merge info
    merged_from: list[str] = field(default_factory=list)
    # Optional GPS/device prior evidence populated after aggregation.
    location: dict[str, Any] = field(default_factory=dict)
    existence_prior: dict[str, Any] = field(default_factory=dict)

    @property
    def on_fraction(self) -> float:
        return self.frames_on / max(self.frames_on + self.frames_off, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "status": self.status,
            "avg_brightness": round(self.avg_brightness, 2),
            "median_brightness": round(self.median_brightness, 2),
            "brightness_std": round(self.brightness_std, 2),
            "min_brightness": round(self.min_brightness, 2),
            "max_brightness": round(self.max_brightness, 2),
            "peak_brightness_mean": round(self.peak_brightness_mean, 2),
            "total_flux_mean": round(self.total_flux_mean, 2),
            "confidence": round(self.confidence, 4),
            "representative_bbox": [round(v, 1) for v in self.representative_bbox],
            "frame_count": self.frame_count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames_on": self.frames_on,
            "frames_off": self.frames_off,
            "on_fraction": round(self.on_fraction, 4),
            "merged_from": self.merged_from,
            "location": self.location,
            "existence_prior": self.existence_prior,
        }


def _detect_flicker(
    on_off_sequence: list[bool],
    threshold_crossings: int,
) -> bool:
    """Return True if the on/off state crosses the threshold more than
    *threshold_crossings* times, indicating a flickering lamp."""
    if len(on_off_sequence) < 3:
        return False
    crossings = sum(
        1
        for a, b in zip(on_off_sequence, on_off_sequence[1:])
        if a != b
    )
    return crossings >= threshold_crossings


def aggregate_measurements(
    measurements: list[LampMeasurement],
    agg_cfg: AggregationConfig,
    meas_cfg: MeasurementConfig,
    merge_map: dict[str, str] | None = None,
) -> list[AggregatedLamp]:
    """Group per-frame measurements by track and compute per-lamp stats.

    Parameters
    ----------
    measurements : list[LampMeasurement]
        All per-frame measurements from the pipeline.
    agg_cfg : AggregationConfig
        Temporal aggregation configuration.
    meas_cfg : MeasurementConfig
        Measurement engine configuration (for brightness threshold).
    merge_map : dict[str, str] | None
        Optional mapping from absorbed track IDs to canonical track IDs
        (from duplicate removal).

    Returns
    -------
    list[AggregatedLamp]
        One entry per unique physical lamp.
    """
    # Apply merge map: redirect absorbed tracks to their canonical ID
    if merge_map:
        for m in measurements:
            m.track_id = merge_map.get(m.track_id, m.track_id)

    # Group by track ID
    groups: dict[str, list[LampMeasurement]] = {}
    for m in measurements:
        groups.setdefault(m.track_id, []).append(m)

    # Sort each group by frame index
    for tid in groups:
        groups[tid].sort(key=lambda x: x.frame_index)

    results: list[AggregatedLamp] = []

    for tid, track_measurements in sorted(groups.items()):
        # Persistence check: discard tracks shorter than min_track_frames
        if len(track_measurements) < agg_cfg.min_track_frames:
            continue

        brightnesses = [m.mean_brightness for m in track_measurements]
        peaks = [m.peak_brightness for m in track_measurements]
        fluxes = [m.total_flux for m in track_measurements]
        on_off = [m.is_on for m in track_measurements]
        confs = [m.detection_confidence for m in track_measurements]

        frames_on = sum(on_off)
        frames_off = len(on_off) - frames_on
        on_fraction = frames_on / max(len(on_off), 1)

        # Flicker detection
        is_flickering = _detect_flicker(on_off, meas_cfg.flicker_crossing_threshold)

        # Decide final status
        if is_flickering:
            status = "flickering"
        elif on_fraction >= agg_cfg.working_frame_fraction:
            status = "working"
        else:
            status = "off"

        # Representative bbox: from highest-confidence frame
        best = max(track_measurements, key=lambda m: m.detection_confidence)

        # Compute merged_from list for merge info
        merged_from: list[str] = []
        if merge_map:
            merged_from = [k for k, v in merge_map.items() if v == tid]

        lamp = AggregatedLamp(
            track_id=tid,
            status=status,
            avg_brightness=statistics.mean(brightnesses),
            median_brightness=statistics.median(brightnesses),
            brightness_std=statistics.stdev(brightnesses) if len(brightnesses) > 1 else 0.0,
            min_brightness=min(brightnesses),
            max_brightness=max(brightnesses),
            peak_brightness_mean=statistics.mean(peaks),
            total_flux_mean=statistics.mean(fluxes),
            confidence=statistics.mean(confs),
            representative_bbox=best.xyxy,
            frame_count=len(track_measurements),
            first_frame=track_measurements[0].frame_index,
            last_frame=track_measurements[-1].frame_index,
            frames_on=frames_on,
            frames_off=frames_off,
            merged_from=merged_from,
        )
        results.append(lamp)

    return results
