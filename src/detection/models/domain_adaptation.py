from __future__ import annotations

from pathlib import Path
from typing import Any

from rbccps_od.models.errors import UnsupportedAdvancedStageError


class DomainAdaptationAdapter:
    def __init__(self, checkpoint: str | Path | None = None, enabled: bool = False) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.enabled = enabled

    def adapt(self, detection_kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.enabled:
            raise UnsupportedAdvancedStageError(
                "Domain adaptation is enabled, but the exact official DAI-Net assets are not streetlight-ready. "
                "The stage is pinned but intentionally disabled by default until a streetlight-specific adaptation path exists."
            )
        return dict(detection_kwargs)
