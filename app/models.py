from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    video_id: str | None = None
    video_url: str | None = None
    fps_target: float | None = None


class VideoInfo(BaseModel):
    video_id: str
    width: int
    height: int
    fps: float


class SessionCreateResponse(BaseModel):
    session_id: str
    video: VideoInfo
    ws_url: str
    fps_target: float


class AnnotationRequest(BaseModel):
    points: list[tuple[float, float]] = Field(default_factory=list)


class AnnotationResponse(BaseModel):
    annotation_id: str


class TrackUpdate(BaseModel):
    session_id: str
    annotation_id: str
    t: float
    points: list[tuple[float, float]]
    confidence: float
    mode: str
    events: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    total_tracks: int
    avg_confidence: float
    reanchor_success: int
    reanchor_fail: int
    out_of_frame: int
    low_confidence: int
    last_mode: str | None = None


class VideoUploadResponse(BaseModel):
    video_id: str
    url: str
