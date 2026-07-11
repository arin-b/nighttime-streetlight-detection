from __future__ import annotations

from pathlib import Path

from rbccps_od.models.enhancer import LowLightEnhancer


class EnhancementStage:
    def __init__(self, enhancer: LowLightEnhancer | None = None, enabled: bool = False) -> None:
        self.enhancer = enhancer or LowLightEnhancer(enabled=enabled)
        self.enabled = enabled

    def run(self, frame_path: str | Path, output_path: str | Path | None = None, device: str | None = None) -> str:
        if not self.enabled:
            return str(frame_path)
        return self.enhancer.enhance(frame_path, output_path=output_path, device=device)
