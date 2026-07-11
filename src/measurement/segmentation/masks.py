from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionMix:
    road: float
    footpath: float
    crossing: float
    verge: float

    def to_dict(self) -> dict[str, float]:
        return {
            "road": round(self.road, 4),
            "footpath": round(self.footpath, 4),
            "crossing": round(self.crossing, 4),
            "verge": round(self.verge, 4),
        }


def estimate_region_mix(frame_width: int, bbox_center_x: float) -> RegionMix:
    """Deterministic geometry-free prior until a trained segmenter is installed."""
    x_norm = max(0.0, min(1.0, bbox_center_x / max(1, frame_width)))
    footpath = 0.22 + 0.18 * abs(x_norm - 0.5) * 2
    road = 0.68 - 0.12 * abs(x_norm - 0.5) * 2
    crossing = 0.03
    verge = max(0.0, 1.0 - road - footpath - crossing)
    return RegionMix(road=road, footpath=footpath, crossing=crossing, verge=verge)
