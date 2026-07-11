from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

# def precision(tp: int, fp: int) -> float:
#     denom = tp + fp
#     return 0.0 if denom == 0 else tp / denom


# def recall(tp: int, fn: int) -> float:
#     denom = tp + fn
#     return 0.0 if denom == 0 else tp / denom

@dataclass
class FrameMetrics:

    tp: int = 0
    fp: int = 0
    fn: int = 0

    detections: int = 0
    accepted: int = 0

    id_switches: int = 0
    duplicate_tracks: int = 0


@dataclass
class TrackMetrics:

    visible_frames: int = 0
    accepted_frames: int = 0
    rejected_frames: int = 0


@dataclass
class MetricsManager:

    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0

    total_detections: int = 0
    total_accepted: int = 0

    total_id_switches: int = 0
    total_duplicate_tracks: int = 0

    total_frames: int = 0

    track_stats: dict = field(default_factory=lambda: defaultdict(TrackMetrics))

    previous_assignments: dict = field(default_factory=dict)

    def update_frame(
        self,
        gt_count: int,
        accepted_tracks: list,
        raw_tracks: list,
    ) -> None:

        self.total_frames += 1

        accepted_count = len(accepted_tracks)
        raw_count = len(raw_tracks)

        tp = min(gt_count, accepted_count)

        fp = max(0, accepted_count - gt_count)

        fn = max(0, gt_count - accepted_count)

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        self.total_detections += raw_count
        self.total_accepted += accepted_count

    def update_track_metrics(
        self,
        track_id: str,
        accepted: bool,
    ) -> None:

        track = self.track_stats[track_id]

        track.visible_frames += 1

        if accepted:
            track.accepted_frames += 1
        else:
            track.rejected_frames += 1

    def update_id_switches(
        self,
        object_key: str,
        current_track_id: str,
    ) -> None:

        previous = self.previous_assignments.get(object_key)

        if previous is not None and previous != current_track_id:
            self.total_id_switches += 1

        self.previous_assignments[object_key] = current_track_id

    @property
    def precision(self) -> float:

        denom = self.total_tp + self.total_fp

        if denom == 0:
            return 0.0

        return self.total_tp / denom

    @property
    def recall(self) -> float:

        denom = self.total_tp + self.total_fn

        if denom == 0:
            return 0.0

        return self.total_tp / denom
    
    @property
    def f1(self) -> float:

        p = self.precision
        r = self.recall

        denom = p + r

        if denom == 0:
            return 0.0

        return 2 * p * r / denom

    @property
    def false_positives_per_frame(self) -> float:

        if self.total_frames == 0:
            return 0.0

        return self.total_fp / self.total_frames

    @property
    def mota(self) -> float:

        denom = self.total_tp + self.total_fn

        if denom == 0:
            return 0.0

        errors = (
            self.total_fn
            + self.total_fp
            + self.total_id_switches
        )

        return 1.0 - (errors / denom)

    def summary(self) -> dict:

        return {

            "frames":

                self.total_frames,

            "tp":

                self.total_tp,

            "fp":

                self.total_fp,

            "fn":

                self.total_fn,

            "precision":

                round(self.precision, 4),

            "recall":

                round(self.recall, 4),

            "f1":

                round(self.f1, 4),

            "mota":

                round(self.mota, 4),

            "id_switches":

                self.total_id_switches,

            "false_positives_per_frame":

                round(self.false_positives_per_frame, 4),

            "total_detections":

                self.total_detections,

            "total_accepted":

                self.total_accepted,
        }
