from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO


class UltralyticsAdapter:
    """Thin wrapper around Ultralytics YOLO for load, predict, validate, and track."""

    def load(self, model_path: str | Path) -> YOLO:
        return YOLO(str(model_path))

    def predict(self, model_path: str | Path, **kwargs: Any) -> Any:
        return self.load(model_path).predict(**kwargs)

    def validate(self, model_path: str | Path, **kwargs: Any) -> Any:
        return self.load(model_path).val(**kwargs)

    def track(self, model_path: str | Path, **kwargs: Any) -> Any:
        return self.load(model_path).track(**kwargs)
