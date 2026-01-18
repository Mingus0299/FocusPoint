from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class LKParams:
    # Feature detection (Shi–Tomasi)
    max_corners: int = 350
    quality_level: float = 0.005
    min_distance: int = 5
    block_size: int = 7

    # LK optical flow
    win_size: Tuple[int, int] = (31, 31)  # larger window helps echo speckle
    max_level: int = 4
    term_criteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )

    # ROI / masks
    pad: int = 18
    ring_px: int = 10  # boundary ring thickness

    # Stabilization / thresholds
    fb_thresh: float = 2.5
    max_step_px: float = 25.0
    min_valid_pts: int = 25
    low_conf_thresh: float = 0.35
    recovered_conf_thresh: float = 0.55

    # Affine estimation
    ransac_thresh: float = 3.0
    min_inliers: int = 10

    # Optional smoothing of affine (small, to reduce jitter)
    smooth_affine_alpha: float = 0.6  # closer to 1.0 = smoother


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _preprocess_echo(gray: np.ndarray) -> np.ndarray:
    # Reduce speckle a bit + improve local contrast
    g = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    return g


def _poly_to_bbox(poly: np.ndarray, pad: int, w: int, h: int) -> Tuple[int, int, int, int]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    x0 = int(np.floor(xs.min())) - pad
    y0 = int(np.floor(ys.min())) - pad
    x1 = int(np.ceil(xs.max())) + pad
    y1 = int(np.ceil(ys.max())) + pad

    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    return x0, y0, x1, y1


def _make_ring_mask(gray: np.ndarray, poly: np.ndarray, ring_px: int) -> np.ndarray:
    """Mask that covers a ring around polygon boundary (good for cavity borders)."""
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

    k = max(1, int(ring_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dil = cv2.dilate(mask, kernel, iterations=1)
    ero = cv2.erode(mask, kernel, iterations=1)
    ring = cv2.subtract(dil, ero)
    return ring


def _seed_points_in_ring(gray: np.ndarray, poly: np.ndarray, bbox, params: LKParams) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    roi = gray[y0:y1 + 1, x0:x1 + 1]
    if roi.size == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    ring = _make_ring_mask(gray, poly, params.ring_px)
    roi_mask = ring[y0:y1 + 1, x0:x1 + 1]
    if roi_mask.size == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=params.max_corners,
        qualityLevel=params.quality_level,
        minDistance=params.min_distance,
        blockSize=params.block_size,
        mask=roi_mask
    )
    if corners is None:
        return np.empty((0, 1, 2), dtype=np.float32)

    corners = corners.astype(np.float32)
    corners[:, 0, 0] += x0
    corners[:, 0, 1] += y0
    return corners


def _forward_backward_filter(prev_gray, gray, p0, params: LKParams):
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None,
        winSize=params.win_size,
        maxLevel=params.max_level,
        criteria=params.term_criteria,
    )
    if p1 is None or st1 is None:
        return np.empty((0, 1, 2), np.float32), np.empty((0, 1, 2), np.float32)

    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, p1, None,
        winSize=params.win_size,
        maxLevel=params.max_level,
        criteria=params.term_criteria,
    )
    if p0_back is None or st2 is None:
        return np.empty((0, 1, 2), np.float32), np.empty((0, 1, 2), np.float32)

    st1 = st1.reshape(-1).astype(bool)
    st2 = st2.reshape(-1).astype(bool)
    fb = np.linalg.norm(p0[:, 0, :] - p0_back[:, 0, :], axis=1)
    good = st1 & st2 & (fb < params.fb_thresh)

    return p0[good].reshape(-1, 1, 2).astype(np.float32), p1[good].reshape(-1, 1, 2).astype(np.float32)


def _apply_affine_to_poly(poly: np.ndarray, M: np.ndarray) -> np.ndarray:
    poly_h = np.hstack([poly, np.ones((poly.shape[0], 1), dtype=np.float32)])  # (N,3)
    new_poly = (M @ poly_h.T).T  # (N,2)
    return new_poly.astype(np.float32)


def _blend_affine(M_prev: Optional[np.ndarray], M_new: np.ndarray, alpha: float) -> np.ndarray:
    """Simple elementwise smoothing of 2x3 affine."""
    if M_prev is None:
        return M_new
    a = float(alpha)
    return (a * M_new + (1.0 - a) * M_prev).astype(np.float32)


class LKTracker:
    """
    Echo-friendly tracker:
      - seeds features on polygon boundary ring
      - LK + forward-backward filtering
      - estimates affine each frame with RANSAC (handles scale/rotation)
      - falls back to translation if affine fails

    Contract:
      init(frame, polygon_points)
      update(frame) -> {"points": [...], "confidence": float, "mode": str, "events": [str]}
    """

    def __init__(self, params: Optional[LKParams] = None) -> None:
        self.p = params or LKParams()
        self._initialized = False

        self._poly: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None

        self._was_low = False
        self._M_smooth: Optional[np.ndarray] = None  # for affine smoothing

    def init(self, frame, polygon_points: Iterable[Point]) -> None:
        gray0 = _preprocess_echo(_to_gray(np.asarray(frame)))

        poly = np.array(list(polygon_points), dtype=np.float32)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            self._initialized = False
            self._poly = None
            self._prev_gray = None
            self._prev_pts = None
            return

        h, w = gray0.shape[:2]
        bbox = _poly_to_bbox(poly, self.p.pad, w, h)
        pts = _seed_points_in_ring(gray0, poly, bbox, self.p)

        self._poly = poly
        self._prev_gray = gray0
        self._prev_pts = pts
        self._was_low = False
        self._M_smooth = None
        self._initialized = True

    def _reseed(self, gray: np.ndarray, events: List[str]) -> None:
        if self._poly is None:
            return
        h, w = gray.shape[:2]
        bbox = _poly_to_bbox(self._poly, self.p.pad, w, h)
        self._prev_pts = _seed_points_in_ring(gray, self._poly, bbox, self.p)
        self._M_smooth = None
        events.append("RESEED_FEATURES")

    def update(self, frame) -> Dict[str, Any]:
        if not self._initialized or self._poly is None or self._prev_gray is None or self._prev_pts is None:
            return {"points": [], "confidence": 0.0, "mode": "idle", "events": ["NOT_INITIALIZED"]}

        gray = _preprocess_echo(_to_gray(np.asarray(frame)))

        events: List[str] = []
        mode = "flow"

        if self._prev_pts.shape[0] == 0:
            events.append("NO_FEATURES")
            self._reseed(gray, events)
            self._prev_gray = gray
            return {"points": [(float(x), float(y)) for x, y in self._poly], "confidence": 0.0, "mode": "idle", "events": events}

        p0_good, p1_good = _forward_backward_filter(self._prev_gray, gray, self._prev_pts, self.p)
        valid = int(p1_good.shape[0])
        total = max(1, int(self._prev_pts.shape[0]))
        conf = float(valid) / float(total)

        if conf < self.p.low_conf_thresh and not self._was_low:
            events.append("TRACKING_LOW_CONFIDENCE")
            self._was_low = True
        elif conf > self.p.recovered_conf_thresh and self._was_low:
            events.append("TRACKING_RECOVERED")
            self._was_low = False

        if valid < 6:
            events.append("TOO_FEW_VALID_PTS")
            self._reseed(gray, events)
            self._prev_gray = gray
            return {"points": [(float(x), float(y)) for x, y in self._poly], "confidence": min(conf, 0.2), "mode": mode, "events": events}

        src = p0_good[:, 0, :]
        dst = p1_good[:, 0, :]

        # Remove crazy jumps before model fit
        d = dst - src
        mag = np.linalg.norm(d, axis=1)
        keep = mag < self.p.max_step_px
        src = src[keep]
        dst = dst[keep]

        applied_affine = False
        if src.shape[0] >= 6:
            M, inliers = cv2.estimateAffinePartial2D(
                src, dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.p.ransac_thresh,
                confidence=0.99,
                maxIters=2000
            )
            if M is not None and inliers is not None and int(inliers.sum()) >= self.p.min_inliers:
                M = M.astype(np.float32)
                M = _blend_affine(self._M_smooth, M, self.p.smooth_affine_alpha)
                self._M_smooth = M
                self._poly = _apply_affine_to_poly(self._poly, M)
                mode = "affine"
                events.append("AFFINE_RANSAC")
                applied_affine = True

        if not applied_affine:
            # Fallback to robust translation
            if src.shape[0] > 0:
                dx = float(np.median(dst[:, 0] - src[:, 0]))
                dy = float(np.median(dst[:, 1] - src[:, 1]))
            else:
                dx, dy = 0.0, 0.0
            self._poly[:, 0] += dx
            self._poly[:, 1] += dy
            events.append("FALLBACK_TRANSLATION")

        # Maintain points
        if valid < self.p.min_valid_pts:
            self._reseed(gray, events)
        else:
            self._prev_pts = p1_good.reshape(-1, 1, 2).astype(np.float32)

        self._prev_gray = gray

        # Out-of-frame
        h, w = gray.shape[:2]
        if (
            np.any(self._poly[:, 0] < 0) or np.any(self._poly[:, 0] > (w - 1)) or
            np.any(self._poly[:, 1] < 0) or np.any(self._poly[:, 1] > (h - 1))
        ):
            events.append("OUT_OF_FRAME")

        return {
            "points": [(float(x), float(y)) for x, y in self._poly],
            "confidence": float(max(0.0, min(1.0, conf))),
            "mode": mode,
            "events": events,
        }
