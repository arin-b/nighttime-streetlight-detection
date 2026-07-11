from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PairedInputFrame:
    dark_frame: Path
    light_frame: Path | None = None
