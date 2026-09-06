"""Optional overlays for inspecting custom-tracker associations."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .custom_tracker import TrackedDetection


def draw_tracks(frame: np.ndarray, tracks: Sequence[TrackedDetection], *, debug: bool = False) -> np.ndarray:
    """Draw current detection boxes, IDs, and optional predicted centres."""

    import cv2

    output = np.asarray(frame).copy()
    for track in tracks:
        cx, cy, width, height = track.bbox
        x1, y1 = int(round(cx - width / 2)), int(round(cy - height / 2))
        x2, y2 = int(round(cx + width / 2)), int(round(cy + height / 2))
        cv2.rectangle(output, (x1, y1), (x2, y2), (70, 220, 70), 2)
        cv2.putText(output, f"ID {track.track_id}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 220, 70), 1, cv2.LINE_AA)
        if debug:
            px, py = (int(round(value)) for value in track.predicted_centre)
            cv2.drawMarker(output, (px, py), (0, 220, 255), cv2.MARKER_CROSS, 10, 1)
            if track.association_cost is not None:
                cv2.putText(output, f"cost {track.association_cost:.1f}", (x1, min(output.shape[0] - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
    return output
