from __future__ import annotations

from typing import Iterable


class HybridTracker:
    """Fallback tracker stub so the backend can run before the LK/ORB tracker lands."""

    def __init__(self) -> None:
        self._points: list[tuple[float, float]] | None = None

    def init(self, frame, polygon_points: Iterable[tuple[float, float]]) -> None:
        self._points = list(polygon_points)

    def update(self, frame) -> dict:
        if not self._points:
            return {
                "points": [],
                "confidence": 0.0,
                "mode": "idle",
                "events": ["NOT_INITIALIZED"],
            }
        return {
            "points": list(self._points),
            "confidence": 0.9,
            "mode": "flow",
            "events": [],
        }
