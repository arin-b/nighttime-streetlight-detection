from __future__ import annotations

from pathlib import Path

from rbccps_od.models.retinex import RetinexDecompositionModel


class DecompositionStage:
    def __init__(self, model: RetinexDecompositionModel | None = None, enabled: bool = False) -> None:
        self.model = model or RetinexDecompositionModel(enabled=enabled)
        self.enabled = enabled

    def run(self, frame_path: str | Path, output_dir: str | Path | None = None, device: str | None = None) -> dict[str, str]:
        if not self.enabled:
            return {"reflectance": str(frame_path), "illumination": str(frame_path)}
        return self.model.decompose(frame_path, output_dir=output_dir, device=device)
