from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Track:
    track_id: str
    bbox: list[float]
    score: float
    age: int = 0
    hits: int = 1
    history: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
