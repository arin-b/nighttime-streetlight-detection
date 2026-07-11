from __future__ import annotations

from rbccps_od.domain.detections import Detection
from rbccps_od.domain.tracks import Track
from rbccps_od.models.tracker import SimpleTracker


class TrackingStage:
    def __init__(self, tracker: SimpleTracker | None = None, enabled: bool = False) -> None:
        self.tracker = tracker or SimpleTracker(enabled=enabled)
        self.enabled = enabled

    def run(self, detections: list[Detection]) -> list[Track]:
        if not self.enabled:
            return [Track(track_id=f"det_{idx}", bbox=det.bbox, score=det.score, history=[det.bbox]) for idx, det in enumerate(detections, start=1)]
        return self.tracker.update(detections)
