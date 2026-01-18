from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class LKParams:
    # --- Feature detection (Shi–Tomasi) ---
    max_corners: int = 600
    quality_level: float = 0.005
    min_distance: int = 5
    block_size: int = 7

    # --- LK optical flow ---
    # Larger window / more pyramid levels helps with echo + faster motion
    win_size: Tuple[int, int] = (41, 41)
    max_level: int = 5
    term_criteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )

    # --- ROI / masks ---
    pad: int = 18
    ring_px: int = 14  # boundary ring thickness (important for cavities)

    # --- Filtering / thresholds ---
    fb_thresh: float = 3.0            # forward-backward error threshold (px)
    max_step_px: float = 50.0         # filter out insane per-point jumps
    min_valid_pts: int = 25
    low_conf_thresh: float = 0.35
    recovered_conf_thresh: float = 0.55

    # --- Affine estimation ---
    ransac_thresh: float = 4.0
    min_inliers: int = 10
    smooth_affine_alpha: float = 0.7  # matrix smoothing (0..1; higher=smoother)
    max_affine_translation_px: float = 45.0  # clamp per-frame tx/ty spike

    # --- Output smoothing (polygon EMA) ---
    poly_smooth_alpha: float = 0.87  # 0.85-0.95 (higher=smoother)

    # --- Recovery (template matching) ---
    template_update_every: int = 15
    template_update_conf: float = 0.75
    template_min_size: int = 20

    # Stage 1: local search (fast)
    local_search_scale: float = 3.0
    local_match_thresh: float = 0.55

    # Stage 2: global search downsampled (robust)
    global_downsample: float = 0.5
    global_match_thresh: float = 0.50

    # Recovery control
    recovery_cooldown_frames: int = 5
    attempt_recovery_on_fallback: bool = True


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _preprocess_echo(gray: np.ndarray) -> np.ndarray:
    # Reduce speckle + boost local contrast (helps low-quality echo)
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


def _bbox_expand_around_center(bbox, scale: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    nw = int(bw * scale)
    nh = int(bh * scale)
    nx0 = int(round(cx - nw / 2))
    ny0 = int(round(cy - nh / 2))
    nx1 = nx0 + nw - 1
    ny1 = ny0 + nh - 1

    nx0 = max(0, min(w - 1, nx0))
    ny0 = max(0, min(h - 1, ny0))
    nx1 = max(0, min(w - 1, nx1))
    ny1 = max(0, min(h - 1, ny1))
    return nx0, ny0, nx1, ny1


def _make_ring_mask(gray: np.ndarray, poly: np.ndarray, ring_px: int) -> np.ndarray:
    """Binary ring mask around polygon boundary (better than cavity interior)."""
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
    if M_prev is None:
        return M_new.astype(np.float32)
    a = float(alpha)
    return (a * M_new + (1.0 - a) * M_prev).astype(np.float32)


def _clamp_affine_translation(M: np.ndarray, max_t: float) -> np.ndarray:
    M = M.astype(np.float32, copy=True)
    M[0, 2] = float(np.clip(M[0, 2], -max_t, max_t))
    M[1, 2] = float(np.clip(M[1, 2], -max_t, max_t))
    return M


def _poly_ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.copy()
    a = float(alpha)
    return (a * prev + (1.0 - a) * cur).astype(np.float32)


def _safe_crop(gray: np.ndarray, bbox) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return gray[y0:y1 + 1, x0:x1 + 1]


def _match_template(search: np.ndarray, templ: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """Return (score, (x, y)) top-left in search coords."""
    if search.size == 0 or templ.size == 0:
        return 0.0, (0, 0)
    if search.shape[0] < templ.shape[0] or search.shape[1] < templ.shape[1]:
        return 0.0, (0, 0)

    res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


class LKTracker:
    """
    Echo-friendly tracker:
      - boundary ring feature seeding
      - LK + forward-backward filtering
      - affine RANSAC each frame + inlier refinement
      - two-stage template recovery for large jumps
      - polygon EMA smoothing for smooth output

    Contract:
      init(frame, polygon_points)
      update(frame) -> {"points": [...], "confidence": float, "mode": str, "events": [str]}
    """

    def __init__(self, params: Optional[LKParams] = None) -> None:
        self.p = params or LKParams()
        self._initialized = False

        self._poly: Optional[np.ndarray] = None
        self._poly_smooth: Optional[np.ndarray] = None

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None

        self._was_low = False
        self._M_smooth: Optional[np.ndarray] = None

        # Recovery state
        self._template: Optional[np.ndarray] = None
        self._template_bbox: Optional[Tuple[int, int, int, int]] = None
        self._frame_idx: int = 0
        self._last_recovery_attempt: int = -10**9

    def init(self, frame, polygon_points: Iterable[Point]) -> None:
        gray0 = _preprocess_echo(_to_gray(np.asarray(frame)))

        poly = np.array(list(polygon_points), dtype=np.float32)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            self._initialized = False
            self._poly = None
            self._poly_smooth = None
            self._prev_gray = None
            self._prev_pts = None
            return

        h, w = gray0.shape[:2]
        bbox = _poly_to_bbox(poly, self.p.pad, w, h)
        pts = _seed_points_in_ring(gray0, poly, bbox, self.p)

        # Template for recovery (use bbox crop)
        templ = _safe_crop(gray0, bbox)
        if templ.shape[0] >= self.p.template_min_size and templ.shape[1] >= self.p.template_min_size:
            self._template = templ.copy()
            self._template_bbox = bbox
        else:
            self._template = None
            self._template_bbox = None

        self._poly = poly
        self._poly_smooth = poly.copy()
        self._prev_gray = gray0
        self._prev_pts = pts
        self._was_low = False
        self._M_smooth = None
        self._initialized = True
        self._frame_idx = 0
        self._last_recovery_attempt = -10**9

    def _reseed(self, gray: np.ndarray, events: List[str]) -> None:
        if self._poly is None:
            return
        h, w = gray.shape[:2]
        bbox = _poly_to_bbox(self._poly, self.p.pad, w, h)
        self._prev_pts = _seed_points_in_ring(gray, self._poly, bbox, self.p)
        self._M_smooth = None
        events.append("RESEED_FEATURES")

    def _maybe_update_template(self, gray: np.ndarray, conf: float, mode: str, events: List[str]) -> None:
        if self._poly is None:
            return
        if conf < self.p.template_update_conf:
            return
        if mode != "affine":
            return
        if self._frame_idx % self.p.template_update_every != 0:
            return

        h, w = gray.shape[:2]
        bbox = _poly_to_bbox(self._poly, self.p.pad, w, h)
        templ = _safe_crop(gray, bbox)
        if templ.shape[0] >= self.p.template_min_size and templ.shape[1] >= self.p.template_min_size:
            self._template = templ.copy()
            self._template_bbox = bbox
            events.append("TEMPLATE_REFRESH")

    def _attempt_recovery(self, gray: np.ndarray, events: List[str]) -> bool:
        """Two-stage recovery: local window first; if fails, global downsampled whole-frame."""
        if self._poly is None or self._template is None:
            return False

        if (self._frame_idx - self._last_recovery_attempt) < self.p.recovery_cooldown_frames:
            return False
        self._last_recovery_attempt = self._frame_idx

        H, W = gray.shape[:2]
        cur_bbox = _poly_to_bbox(self._poly, self.p.pad, W, H)
        templ = self._template

        # Stage 1: local search
        search_bbox = _bbox_expand_around_center(cur_bbox, self.p.local_search_scale, W, H)
        search = _safe_crop(gray, search_bbox)

        score, (mx, my) = _match_template(search, templ)
        if score >= self.p.local_match_thresh:
            sx0, sy0, _, _ = search_bbox
            new_x0 = sx0 + mx
            new_y0 = sy0 + my

            old_x0, old_y0, _, _ = cur_bbox
            dx = float(new_x0 - old_x0)
            dy = float(new_y0 - old_y0)

            self._poly[:, 0] += dx
            self._poly[:, 1] += dy
            events.append("RECOVER_LOCAL_SUCCESS")
            self._reseed(gray, events)
            return True
        else:
            events.append("RECOVER_LOCAL_FAIL")

        # Stage 2: global downsampled search
        ds = float(self.p.global_downsample)
        if not (0.0 < ds < 1.0):
            ds = 0.5

        small = cv2.resize(gray, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
        templ_s = cv2.resize(templ, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)

        if templ_s.shape[0] < 8 or templ_s.shape[1] < 8:
            events.append("RECOVER_GLOBAL_SKIP_SMALL_TEMPLATE")
            return False

        score_g, (gx, gy) = _match_template(small, templ_s)
        if score_g >= self.p.global_match_thresh:
            new_x0 = int(round(gx / ds))
            new_y0 = int(round(gy / ds))

            old_x0, old_y0, _, _ = cur_bbox
            dx = float(new_x0 - old_x0)
            dy = float(new_y0 - old_y0)

            self._poly[:, 0] += dx
            self._poly[:, 1] += dy
            events.append("RECOVER_GLOBAL_SUCCESS")
            self._reseed(gray, events)
            return True
        else:
            events.append("RECOVER_GLOBAL_FAIL")

        return False

    def update(self, frame) -> Dict[str, Any]:
        if not self._initialized or self._poly is None or self._prev_gray is None or self._prev_pts is None:
            return {"points": [], "confidence": 0.0, "mode": "idle", "events": ["NOT_INITIALIZED"]}

        self._frame_idx += 1
        gray = _preprocess_echo(_to_gray(np.asarray(frame)))

        events: List[str] = []
        mode = "flow"

        if self._prev_pts.shape[0] == 0:
            events.append("NO_FEATURES")
            self._attempt_recovery(gray, events)
            self._prev_gray = gray
            self._poly_smooth = _poly_ema(self._poly_smooth, self._poly, self.p.poly_smooth_alpha)
            return {
                "points": [(float(x), float(y)) for x, y in self._poly_smooth],
                "confidence": 0.0,
                "mode": "idle",
                "events": events,
            }

        p0_good, p1_good = _forward_backward_filter(self._prev_gray, gray, self._prev_pts, self.p)
        valid = int(p1_good.shape[0])
        total = max(1, int(self._prev_pts.shape[0]))
        conf = float(valid) / float(total)

        # Lost / recovered events
        if conf < self.p.low_conf_thresh and not self._was_low:
            events.append("TRACKING_LOW_CONFIDENCE")
            self._was_low = True
        elif conf > self.p.recovered_conf_thresh and self._was_low:
            events.append("TRACKING_RECOVERED")
            self._was_low = False

        if valid < 6:
            events.append("TOO_FEW_VALID_PTS")
            recovered = self._attempt_recovery(gray, events)
            if not recovered:
                self._reseed(gray, events)
            self._prev_gray = gray
            self._poly_smooth = _poly_ema(self._poly_smooth, self._poly, self.p.poly_smooth_alpha)
            return {
                "points": [(float(x), float(y)) for x, y in self._poly_smooth],
                "confidence": min(conf, 0.2),
                "mode": mode,
                "events": events,
            }

        src = p0_good[:, 0, :]
        dst = p1_good[:, 0, :]

        # Filter absurd point jumps before fitting model
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
                maxIters=2000,
            )

            if M is not None and inliers is not None:
                inlier_count = int(inliers.sum())
                if inlier_count >= self.p.min_inliers:
                    # Refine affine with least-squares on inliers (more precise)
                    inl = inliers.ravel().astype(bool)
                    # Refine affine with least-squares on inliers (more precise)
                    try:
                        M_refined, _ = cv2.estimateAffinePartial2D(src[inl], dst[inl])
                        if M_refined is not None:
                            M = M_refined
                    except cv2.error:
                        # If this OpenCV build is picky, just keep the RANSAC result
                        pass

                    if M_refined is not None:
                        M = M_refined

                    M = M.astype(np.float32)
                    M = _clamp_affine_translation(M, self.p.max_affine_translation_px)
                    M = _blend_affine(self._M_smooth, M, self.p.smooth_affine_alpha)
                    self._M_smooth = M

                    self._poly = _apply_affine_to_poly(self._poly, M)
                    mode = "affine"
                    events.append("AFFINE_RANSAC")
                    applied_affine = True

        if not applied_affine:
            # Fallback translation
            if src.shape[0] > 0:
                dx = float(np.median(dst[:, 0] - src[:, 0]))
                dy = float(np.median(dst[:, 1] - src[:, 1]))
            else:
                dx, dy = 0.0, 0.0
            self._poly[:, 0] += dx
            self._poly[:, 1] += dy
            events.append("FALLBACK_TRANSLATION")

            if self.p.attempt_recovery_on_fallback:
                self._attempt_recovery(gray, events)

        # Maintain points
        if valid < self.p.min_valid_pts:
            self._reseed(gray, events)
        else:
            self._prev_pts = p1_good.reshape(-1, 1, 2).astype(np.float32)

        self._prev_gray = gray

        # Template refresh when stable
        self._maybe_update_template(gray, conf, mode, events)

        # Smooth polygon output (final)
        self._poly_smooth = _poly_ema(self._poly_smooth, self._poly, self.p.poly_smooth_alpha)

        # Out-of-frame event
        h, w = gray.shape[:2]
        if (
            np.any(self._poly_smooth[:, 0] < 0) or np.any(self._poly_smooth[:, 0] > (w - 1)) or
            np.any(self._poly_smooth[:, 1] < 0) or np.any(self._poly_smooth[:, 1] > (h - 1))
        ):
            events.append("OUT_OF_FRAME")

        return {
            "points": [(float(x), float(y)) for x, y in self._poly_smooth],
            "confidence": float(max(0.0, min(1.0, conf))),
            "mode": mode,
            "events": events,
        }
