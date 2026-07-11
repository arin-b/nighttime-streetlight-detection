from __future__ import annotations

from rbccps_od.domain.cues import CueScore


def weighted_aggregate(cues: list[CueScore]) -> float:
    if not cues:
        return 0.0
    weight_sum = sum(max(cue.weight, 0.0) for cue in cues)
    if weight_sum <= 0:
        return 0.0
    return sum(cue.value * cue.weight for cue in cues) / weight_sum
