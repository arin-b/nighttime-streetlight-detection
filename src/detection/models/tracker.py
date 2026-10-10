from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rbccps_od.domain.detections import Detection
from rbccps_od.domain.tracks import Track
from rbccps_od.models.errors import UnsupportedAdvancedStageError


class SimpleTracker:
    def __init__(self, checkpoint: str | Path | None = None, enabled: bool = False) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.enabled = enabled
        self._next_id = 1
        self._tracks: list[Track] = []

    def compensate_camera_motion(self, tracks: list[Track]) -> list[Track]:
        return list(tracks)

    def update(self, detections: list[Detection]) -> list[Track]:
        if self.enabled:
            raise UnsupportedAdvancedStageError(
                "Tracking is enabled, but no exact OSS tracker implementation/checkpoint "
                "has been pinned in the model registry yet."
            )
        if not self._tracks:
            self._tracks = [
                Track(track_id=f"track_{idx}", bbox=det.bbox, score=det.score, history=[det.bbox], metadata={"source": det.source})
                for idx, det in enumerate(detections, start=self._next_id)
            ]
            self._next_id += len(detections)
            return list(self._tracks)

        updated: list[Track] = []
        compensated = self.compensate_camera_motion(self._tracks)
        for existing, det in zip(compensated, detections):
            updated.append(
                replace(
                    existing,
                    bbox=det.bbox,
                    score=det.score,
                    age=existing.age + 1,
                    hits=existing.hits + 1,
                    history=existing.history + [det.bbox],
                )
            )
        if len(detections) > len(updated):
            for det in detections[len(updated):]:
                updated.append(Track(track_id=f"track_{self._next_id}", bbox=det.bbox, score=det.score, history=[det.bbox], metadata={"source": det.source}))
                self._next_id += 1
        self._tracks = updated
        return list(self._tracks)
