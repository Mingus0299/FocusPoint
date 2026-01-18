import mimetypes
import os
import time
import uuid
from urllib.parse import urlparse

import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.config import settings
from app.db import connect, disconnect, get_db
from app.models import (
    AnnotationRequest,
    AnnotationResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionSummary,
    VideoInfo,
    VideoUploadResponse,
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


def _is_url(value: str) -> bool:
    try:
        return urlparse(value).scheme in {"http", "https", "rtsp"}
    except Exception:
        return False


def _video_id_from_url(video_url: str) -> str:
    parsed = urlparse(video_url)
    name = os.path.basename(parsed.path)
    return name or "stream"


def _resolve_default_video_path() -> str:
    path = settings.video_path
    if _is_url(path):
        return path
    if os.path.exists(path):
        return path
    candidate = os.path.join(settings.video_dir, path)
    if os.path.exists(candidate):
        return candidate
    raise HTTPException(status_code=404, detail="Default video not found")


def _resolve_local_video_path(video_id: str) -> str:
    if not video_id:
        raise HTTPException(status_code=400, detail="Missing video_id")
    safe_id = os.path.basename(video_id)
    if safe_id != video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id")
    path = os.path.join(settings.video_dir, safe_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return path


def _resolve_video_source(video_id: str | None, video_url: str | None) -> tuple[str, str]:
    if video_url:
        return video_url, (video_id or _video_id_from_url(video_url))
    if video_id:
        return _resolve_local_video_path(video_id), video_id
    default_path = _resolve_default_video_path()
    if _is_url(default_path):
        return default_path, _video_id_from_url(default_path)
    return default_path, os.path.basename(default_path)


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
    if payload.video_url:
        raise HTTPException(status_code=400, detail="video_url is not supported. Upload a video and use video_id.")
    if not payload.video_id:
        raise HTTPException(status_code=400, detail="video_id is required. Upload a video first.")
    video_id = payload.video_id
    video_path = _resolve_local_video_path(video_id)
    try:
        state = await session_registry.create_session(video_path, fps_target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    video_info = state.video.info()

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
        fps_target=fps_target,
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


@app.post("/sessions/{session_id}/pause")
async def pause_session(session_id: str) -> dict:
    ok = await session_registry.pause_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"paused": True}


@app.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str) -> dict:
    ok = await session_registry.resume_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"paused": False}


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
async def video_info(video_id: str | None = None) -> VideoInfo:
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required")
    path = _resolve_local_video_path(video_id)
    try:
        source = VideoSource(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    info = source.info()
    source.close()
    return VideoInfo(
        video_id=video_id,
        width=info.width,
        height=info.height,
        fps=info.fps,
    )


@app.get("/video/file/{video_id}")
async def video_file(video_id: str):
    path = _resolve_local_video_path(video_id)
    media_type, _ = mimetypes.guess_type(path)
    return FileResponse(path, media_type=media_type or "application/octet-stream", filename=video_id)


@app.post("/video/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    filename = file.filename or "video.mp4"
    _, ext = os.path.splitext(filename)
    ext = ext.lower() if ext else ".mp4"
    video_id = f"{uuid.uuid4().hex}{ext}"

    os.makedirs(settings.video_dir, exist_ok=True)
    dest_path = os.path.join(settings.video_dir, video_id)

    with open(dest_path, "wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    return VideoUploadResponse(video_id=video_id, url=f"/video/file/{video_id}")


@app.get("/video/stream")
async def video_stream(video_id: str | None = None):
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required")
    path = _resolve_local_video_path(video_id)
    try:
        source = VideoSource(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def frame_generator():
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
