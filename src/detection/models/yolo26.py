from __future__ import annotations

from pathlib import Path
from typing import Any

from rbccps_od.models.checkpoint_resolver import resolve_checkpoint
from rbccps_od.training.ultralytics_adapter import UltralyticsAdapter


class YOLO26Detector:
    def __init__(self, asset_name: str = "yolov26_base", checkpoint_path: str | Path | None = None) -> None:
        self.asset_name = asset_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.adapter = UltralyticsAdapter()

    def resolved_checkpoint(self) -> Path:
        if self.checkpoint_path is not None:
            return self.checkpoint_path.resolve()
        checkpoint = resolve_checkpoint(self.asset_name, allow_missing=False)
        assert checkpoint is not None
        return checkpoint

    def predict(self, source: str | Path, **kwargs: Any) -> Any:
        return self.adapter.predict(self.resolved_checkpoint(), source=str(source), **kwargs)

    def track(self, source: str | Path, **kwargs: Any) -> Any:
        return self.adapter.track(self.resolved_checkpoint(), source=str(source), **kwargs)
