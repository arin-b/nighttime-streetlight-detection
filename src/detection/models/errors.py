from __future__ import annotations


class UnsupportedAdvancedStageError(RuntimeError):
    """Raised when an advanced OD stage is enabled without a pinned implementation."""

