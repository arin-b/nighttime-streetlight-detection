"""Lightweight, geometry-only streetlight tracking.

This package is deliberately independent of the detector and the project's
long-term trajectory store.  It only keeps the short-term state needed to
associate detections from adjacent frames.
"""

from .config import TRACKING_MODES, TrackerConfig
from .custom_tracker import CustomStreetlightTracker, TrackedDetection

__all__ = ["TRACKING_MODES", "CustomStreetlightTracker", "TrackedDetection", "TrackerConfig"]
