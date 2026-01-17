from dataclasses import dataclass

import cv2


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float


class VideoSource:
    def __init__(self, path: str) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_index = 0

    def info(self) -> VideoInfo:
        return VideoInfo(width=self.width, height=self.height, fps=self.fps)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        self.frame_index += 1
        return frame, self.frame_index

    def reset(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_index = 0

    def close(self) -> None:
        self.cap.release()
