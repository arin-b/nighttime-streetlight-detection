from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SceneGraph:
    nodes: list[dict[str, object]] = field(default_factory=list)
    edges: list[dict[str, object]] = field(default_factory=list)
