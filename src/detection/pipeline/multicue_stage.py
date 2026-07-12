from __future__ import annotations

from rbccps_od.config.schemas import CueWeights
from rbccps_od.domain.cues import CueScore
from rbccps_od.domain.tracks import Track
from rbccps_od.pipeline.aggregator import weighted_aggregate
from rbccps_od.pipeline.thresholding import threshold_score


def trajectory_cue(track: Track) -> CueScore:
    history_len = len(track.history)
    score = min(1.0, history_len / 4.0)
    return CueScore(name="trajectory", value=score, weight=1.0, metadata={"history_len": history_len})


def size_progression_cue(track: Track) -> CueScore:
    areas = []
    for box in track.history:
        if len(box) >= 4:
            # bbox is typically xywh or similar; w and h are usually at index 2 and 3
            areas.append(max(0.0, box[2]) * max(0.0, box[3]))
            
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


def light_characteristics_cue(track: Track) -> CueScore:
    score = max(0.0, min(1.0, track.score))
    return CueScore(name="light_characteristics", value=score, weight=1.0, metadata={})


def position_prior_cue(track: Track, frame_height: float | None = None) -> CueScore:
    if len(track.bbox) < 4:
        return CueScore(name="position_prior", value=0.0, weight=1.0)
    _, y, _, h = track.bbox
    center_y = y + h / 2.0
    if frame_height and frame_height > 0:
        normalized = center_y / frame_height
        score = 1.0 - abs(normalized - 0.4)
    else:
        score = 0.5
    return CueScore(name="position_prior", value=max(0.0, min(1.0, score)), weight=1.0, metadata={"center_y": center_y})


class MultiCueFilterStage:
    def __init__(self, weights: CueWeights, threshold: float = 0.5, enabled: bool = False) -> None:
        self.weights = weights
        self.threshold = threshold
        self.enabled = enabled

    def score_track(self, track: Track, frame_height: float | None = None) -> tuple[float, list[CueScore]]:
        cues = [
            trajectory_cue(track),
            size_progression_cue(track),
            light_characteristics_cue(track),
            position_prior_cue(track, frame_height=frame_height),
        ]
        for cue in cues:
            cue.weight = getattr(self.weights, cue.name)
        aggregate = weighted_aggregate(cues)
        return aggregate, cues

    def run(self, tracks: list[Track], frame_height: float | None = None) -> list[dict]:
        if not self.enabled:
            return [{"track": track, "aggregate_score": track.score, "accepted": True, "cues": []} for track in tracks]
        results = []
        for track in tracks:
            aggregate, cues = self.score_track(track, frame_height=frame_height)
            results.append({"track": track, "aggregate_score": aggregate, "accepted": threshold_score(aggregate, self.threshold), "cues": cues})
        return results
