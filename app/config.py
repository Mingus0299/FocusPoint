import os
from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value


class Settings:
    def __init__(self) -> None:
        self.mongo_uri: str = _env("MONGO_URI", "mongodb://localhost:27017")
        self.mongo_db: str = _env("MONGO_DB", "focuspoint")
        self.video_path: str = _env("VIDEO_PATH", "videos/echo1.mp4")
        self.video_dir: str = _env("VIDEO_DIR", "videos")
        self.fps_target = float(_env("FPS_TARGET", "30"))
        self.track_write_interval = float(_env("TRACK_WRITE_INTERVAL", "0.2"))
        self.gumloop_webhook_url: str | None = os.getenv("GUMLOOP_WEBHOOK_URL") or None


settings = Settings()
