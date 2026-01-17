import os
import time
import uuid

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.db import connect, disconnect, get_db
from app.models import (
    AnnotationRequest,
    AnnotationResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionSummary,
    VideoInfo,
)
from app.services.gumloop import send_session_complete
from app.services.sessions import SessionRegistry
from app.services.summary import build_summary
from app.video import VideoSource
from app.ws import ConnectionManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_manager = ConnectionManager()
session_registry = SessionRegistry(ws_manager, get_db)


@app.on_event("startup")
async def startup() -> None:
    await connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    await disconnect()


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(payload: SessionCreateRequest) -> SessionCreateResponse:
    fps_target = payload.fps_target or settings.fps_target
    state = await session_registry.create_session(settings.video_path, fps_target)

    video_info = state.video.info()
    video_id = payload.video_id or os.path.basename(settings.video_path)

    db = get_db()
    if db is not None:
        await db.sessions.insert_one(
            {
                "session_id": state.session_id,
                "created_at": time.time(),
                "video_id": video_id,
                "fps_target": fps_target,
            }
        )

    return SessionCreateResponse(
        session_id=state.session_id,
        video=VideoInfo(
            video_id=video_id,
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
        ),
        ws_url=f"/sessions/{state.session_id}/updates",
    )


@app.post("/sessions/{session_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(session_id: str, payload: AnnotationRequest) -> AnnotationResponse:
    annotation_id = str(uuid.uuid4())
    ok = await session_registry.set_annotation(session_id, annotation_id, payload.points)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")

    db = get_db()
    if db is not None:
        await db.annotations.insert_one(
            {
                "session_id": session_id,
                "annotation_id": annotation_id,
                "points": payload.points,
                "created_at": time.time(),
            }
        )

    return AnnotationResponse(annotation_id=annotation_id)


@app.post("/sessions/{session_id}/end", response_model=SessionSummary)
async def end_session(session_id: str) -> SessionSummary:
    state = await session_registry.get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    await session_registry.end_session(session_id)
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    summary = await build_summary(db, session_id)
    await send_session_complete(session_id, summary.dict())
    return summary


@app.get("/sessions/{session_id}/summary", response_model=SessionSummary)
async def get_summary(session_id: str) -> SessionSummary:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    return await build_summary(db, session_id)


@app.websocket("/sessions/{session_id}/updates")
async def updates_ws(websocket: WebSocket, session_id: str) -> None:
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(session_id, websocket)


@app.get("/video/info", response_model=VideoInfo)
async def video_info() -> VideoInfo:
    source = VideoSource(settings.video_path)
    info = source.info()
    source.close()
    return VideoInfo(
        video_id=os.path.basename(settings.video_path),
        width=info.width,
        height=info.height,
        fps=info.fps,
    )


@app.get("/video/stream")
async def video_stream():
    def frame_generator():
        source = VideoSource(settings.video_path)
        try:
            while True:
                frame, _ = source.read()
                if frame is None:
                    source.reset()
                    continue
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                )
                time.sleep(1.0 / source.fps if source.fps > 0 else 0.03)
        finally:
            source.close()

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
