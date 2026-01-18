from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from app.config import settings
from app.models import TrackUpdate
from app.services.gumloop import send_event
from app.tracking.geometry import apply_affine, apply_translation
from app.tracking_v2.tracker_lk import LKTracker
from app.video import VideoSource
from app.ws import ConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class PendingAnnotation:
    annotation_id: str
    points: list[tuple[float, float]]


@dataclass
class SessionState:
    session_id: str
    video_path: str
    fps_target: float
    video: VideoSource
    tracker: LKTracker
    annotation_id: str | None = None
    annotation_points: list[tuple[float, float]] | None = None
    pending_annotation: PendingAnnotation | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_db_write: float = 0.0
    task: asyncio.Task | None = None
    started_at: float = field(default_factory=time.monotonic)


def _normalize_points(points) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


def _apply_tracker_result(
    points: list[tuple[float, float]], result: dict | None
) -> tuple[list[tuple[float, float]], float, str, list[str]]:
    if result is None:
        return points, 0.0, "idle", []
    events = list(result.get("events", []))
    confidence = float(result.get("confidence", 0.0))
    mode = result.get("mode", "flow")

    if "points" in result and result["points"]:
        new_points = _normalize_points(result["points"])
    elif "affine" in result:
        new_points = apply_affine(points, result["affine"])
    elif "dxdy" in result:
        dx, dy = result["dxdy"]
        new_points = apply_translation(points, float(dx), float(dy))
    else:
        new_points = points

    return new_points, confidence, mode, events


async def _store_event(db, session_id: str, annotation_id: str | None, t: float, event_type: str, meta: dict | None = None) -> None:
    if db is None:
        return
    await db.events.insert_one(
        {
            "session_id": session_id,
            "annotation_id": annotation_id,
            "t": t,
            "type": event_type,
            "meta": meta or {},
        }
    )


async def run_tracking_loop(state: SessionState, ws_manager: ConnectionManager, db_getter) -> None:
    db = db_getter()
    frame_interval = 1.0 / state.fps_target if state.fps_target > 0 else 0.0

    while not state.stop_event.is_set():
        loop_started = time.monotonic()
        frame, frame_index = state.video.read()
        if frame is None:
            state.video.reset()
            continue

        if state.pending_annotation:
            pending = state.pending_annotation
            try:
                state.tracker.init(frame, pending.points)
                state.annotation_id = pending.annotation_id
                state.annotation_points = pending.points
                await _store_event(db, state.session_id, pending.annotation_id, 0.0, "ANNOTATION_INIT")
            except Exception as exc:
                logger.exception("Failed to init tracker: %s", exc)
                await _store_event(db, state.session_id, pending.annotation_id, 0.0, "ANNOTATION_INIT_FAIL")
            state.pending_annotation = None

        if state.annotation_points:
            result = state.tracker.update(frame)
            updated_points, confidence, mode, events = _apply_tracker_result(state.annotation_points, result)
            state.annotation_points = updated_points

            t = time.monotonic() - state.started_at
            update = TrackUpdate(
                session_id=state.session_id,
                annotation_id=state.annotation_id or "",
                t=t,
                points=updated_points,
                confidence=confidence,
                mode=mode,
                events=events,
            )
            await ws_manager.broadcast_session(state.session_id, update.dict())

            now = time.monotonic()
            if db is not None and (now - state.last_db_write) >= settings.track_write_interval:
                await db.tracks.insert_one(
                    {
                        "session_id": update.session_id,
                        "annotation_id": update.annotation_id,
                        "t": update.t,
                        "points": update.points,
                        "confidence": update.confidence,
                        "mode": update.mode,
                    }
                )
                state.last_db_write = now

            for event_type in events:
                await _store_event(db, state.session_id, state.annotation_id, t, event_type)
                if event_type in {"LOW_CONFIDENCE", "REANCHOR_SUCCESS", "REANCHOR_FAIL"}:
                    await send_event(event_type, {"session_id": state.session_id, "annotation_id": state.annotation_id, "t": t})

        if frame_interval > 0:
            elapsed = time.monotonic() - loop_started
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)


class SessionRegistry:
    def __init__(self, ws_manager: ConnectionManager, db_getter) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._ws_manager = ws_manager
        self._db_getter = db_getter

    async def create_session(self, video_path: str, fps_target: float) -> SessionState:
        session_id = str(uuid.uuid4())
        video = VideoSource(video_path)
        tracker = LKTracker()
        state = SessionState(
            session_id=session_id,
            video_path=video_path,
            fps_target=fps_target,
            video=video,
            tracker=tracker,
        )
        state.task = asyncio.create_task(run_tracking_loop(state, self._ws_manager, self._db_getter))
        async with self._lock:
            self._sessions[session_id] = state
        return state

    async def end_session(self, session_id: str) -> bool:
        async with self._lock:
            state = self._sessions.get(session_id)
        if not state:
            return False
        state.stop_event.set()
        if state.task:
            await state.task
        state.video.close()
        async with self._lock:
            self._sessions.pop(session_id, None)
        return True

    async def set_annotation(self, session_id: str, annotation_id: str, points: list[tuple[float, float]]) -> bool:
        async with self._lock:
            state = self._sessions.get(session_id)
        if not state:
            return False
        state.pending_annotation = PendingAnnotation(annotation_id=annotation_id, points=points)
        return True

    async def get_session(self, session_id: str) -> SessionState | None:
        async with self._lock:
            return self._sessions.get(session_id)
