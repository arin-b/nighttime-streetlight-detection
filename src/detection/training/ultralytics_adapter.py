from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rbccps_od.config.paths import ensure_dir, repo_root


class UltralyticsAdapter:
    def _import_yolo(self):
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ensure_dir(repo_root() / "_ultralytics_config")))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit("ultralytics is not installed. Install the training extras before running this command.") from exc
        return YOLO

    def load(self, model_path: str | Path):
        YOLO = self._import_yolo()
        return YOLO(str(Path(model_path).resolve()))

    def train(self, model_path: str | Path, **kwargs: Any) -> Any:
        model = self.load(model_path)
        return model.train(**kwargs)

    def validate(self, model_path: str | Path, **kwargs: Any) -> Any:
        model = self.load(model_path)
        return model.val(**kwargs)

    def predict(self, model_path: str | Path, **kwargs: Any) -> Any:
        model = self.load(model_path)
        return model.predict(**kwargs)

    def track(self, model_path: str | Path, **kwargs: Any) -> Any:
        model = self.load(model_path)
        return model.track(**kwargs)
