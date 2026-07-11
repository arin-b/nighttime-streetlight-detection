from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    bbox: list[float]
    score: float
    label: str = "streetlight"
    source: str = "detector"
    metadata: dict[str, Any] = field(default_factory=dict)
