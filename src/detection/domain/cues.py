from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CueScore:
    name: str
    value: float
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)
