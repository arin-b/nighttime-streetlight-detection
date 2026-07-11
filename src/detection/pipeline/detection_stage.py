from __future__ import annotations

from pathlib import Path
from typing import Any

from rbccps_od.models.domain_adaptation import DomainAdaptationAdapter
from rbccps_od.models.yolo26 import YOLO26Detector


class DetectionStage:
    def __init__(
        self,
        detector: YOLO26Detector,
        domain_adapter: DomainAdaptationAdapter | None = None,
        enable_domain_adaptation: bool = False,
    ) -> None:
        self.detector = detector
        self.domain_adapter = domain_adapter or DomainAdaptationAdapter(enabled=enable_domain_adaptation)
        self.enable_domain_adaptation = enable_domain_adaptation

    def run(self, frame_path: str | Path, **kwargs: Any) -> Any:
        payload = dict(kwargs)
        if self.enable_domain_adaptation:
            payload = self.domain_adapter.adapt(payload)
        return self.detector.predict(frame_path, **payload)
