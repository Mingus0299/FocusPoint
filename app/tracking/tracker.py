from __future__ import annotations

from typing import Iterable

from app.tracking_v2.tracker_lk import LKTracker


class HybridTracker:
    """Adapter for the LK tracker implementation from tracking_v2."""

    def __init__(self) -> None:
        self._tracker = LKTracker()

    def init(self, frame, polygon_points: Iterable[tuple[float, float]]) -> None:
        self._tracker.init(frame, polygon_points)

    def update(self, frame) -> dict:
        return self._tracker.update(frame)
