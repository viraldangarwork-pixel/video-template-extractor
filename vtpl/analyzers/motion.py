"""Per-shot camera motion (zoom / pan / static) from optical flow on cached frames."""
import math
from pathlib import Path

import cv2
import numpy as np

from vtpl import frames as fr
from vtpl.util.workdir import WorkDir

OUTPUT = "motion.json"
MIN_POINTS = 20
RANSAC_PX = 2.0            # inlier distance for the per-frame-pair fit
REFIT_PX = 0.5             # tighter, for the fit without static overlays: points stuck on an overlay's edge
                           # move less than a pixel per pair, so 2 px would still count them in
RESEED = 50
ZOOM_MIN = 0.04            # |scale - 1| for a zoom
PAN_MIN = 0.03             # centre movement, share of the frame, for a pan
EDGE_FRAMES = 1            # cached frames skipped at each end of a shot
MAX_K = 3                  # longest stretch of the measured window's motion over the whole shot
MAX_SCALE = 1.5            # total zoom is kept within [1/MAX_SCALE, MAX_SCALE]
MAX_DXY = 0.25             # and pans within this share of the frame on each axis
STATIC = {"type": "static", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.0, "dy": 0.0}


MIN_R2 = 0.6               # how well a straight line must fit the zoom / centre path to count as a move
STILL_PX = 1.0             # a feature that never moves further than this over the window is "static"
STILL_RADIUS = 10          # px around a static feature kept out of the second fit


def _features(gray: np.ndarray, mask: np.ndarray | None = None):
    return cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask)


def static_mask(frames_gray: list[np.ndarray]) -> np.ndarray | None:
    """A feature mask (0 = excluded) around features that stay put over the whole window, such as burnt-in
    text or logos; None if there are too few of them to matter."""
    pts = _features(frames_gray[0])
    if pts is None or len(pts) < MIN_POINTS:
        return None
    origin = pts.reshape(-1, 2).copy()
    alive = np.ones(len(origin), bool)
    far = np.zeros(len(origin))
    cur = pts
    for a, b in zip(frames_gray, frames_gray[1:]):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, cur, None)
        alive &= status.reshape(-1) == 1
        far = np.maximum(far, np.linalg.norm(nxt.reshape(-1, 2) - origin, axis=1))
        cur = nxt
    still = origin[alive & (far < STILL_PX)]
    if len(still) < MIN_POINTS // 2:
        return None
    mask = np.full(frames_gray[0].shape, 255, np.uint8)
    for x, y in still:
        cv2.circle(mask, (int(round(x)), int(round(y))), STILL_RADIUS, 0, -1)
    return mask


def _magnitude(m: tuple[float, float, float]) -> float:
    return abs(math.log(max(m[0], 1e-6))) + math.hypot(m[1], m[2])


def _line_fit(t: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, float]:
    """Slopes of straight lines through each column of ys over t, and the pooled R² of those lines."""
    tc = t - t.mean()
    slopes = (tc @ (ys - ys.mean(axis=0))) / max(float(tc @ tc), 1e-12)
    resid = ys - ys.mean(axis=0) - np.outer(tc, slopes)
    total = float(((ys - ys.mean(axis=0)) ** 2).sum())
    return slopes, (1 - float((resid ** 2).sum()) / total) if total > 1e-12 else 0.0


def steady_motion(series: np.ndarray, n_pairs: float) -> tuple[float, float, float]:
    """(scale, dx, dy) over n_pairs frame pairs, from per-pair steps (rows: pair index, dlog scale, dx, dy).
    Zoom (log scale) and centre position are each fitted with a straight line over time; a path that wanders
    back and forth (handheld shake) fits badly and counts as no motion."""
    t = np.arange(len(series) + 1, dtype=float)      # per tracked pair: lost pairs are extrapolated over
    path = np.vstack([np.zeros(3), np.cumsum(series[:, 1:], axis=0)])
    zoom, r2_zoom = _line_fit(t, path[:, :1])
    shift, r2_shift = _line_fit(t, path[:, 1:])
    log_s = float(zoom[0]) * n_pairs if r2_zoom >= MIN_R2 else 0.0
    dx, dy = (float(v) * n_pairs for v in shift) if r2_shift >= MIN_R2 else (0.0, 0.0)
    return math.exp(log_s), dx, dy


def _best_series(frames_gray: list[np.ndarray]) -> np.ndarray | None:
    """track_series, ignoring static overlays: when the frame moves, features that stay put (burnt-in text)
    pull the fit towards no motion, so fit again without them and keep the larger move if enough is left."""
    n = len(frames_gray) - 1
    first = track_series(frames_gray)
    if first is None or classify(*steady_motion(first, n))["type"] == "static":
        return first
    mask = static_mask(frames_gray)
    second = track_series(frames_gray, mask, REFIT_PX) if mask is not None else None
    if second is not None and _magnitude(steady_motion(second, n)) > _magnitude(steady_motion(first, n)):
        return second
    return first


def window_motion(frames_gray: list[np.ndarray]) -> tuple[float, float, float] | None:
    """Steady camera motion (scale, dx, dy) across the frames, or None if too little can be tracked."""
    series = _best_series(frames_gray)
    return None if series is None else steady_motion(series, len(frames_gray) - 1)


def track_motion(frames_gray: list[np.ndarray], mask: np.ndarray | None = None,
                 ransac_px: float = RANSAC_PX) -> tuple[float, float, float] | None:
    """Total (scale, centre dx, centre dy) across the frames, or None if too little can be tracked. Pairs that
    could not be tracked are assumed to move like the tracked ones."""
    series = track_series(frames_gray, mask, ransac_px)
    if series is None:
        return None
    total = series[:, 1:].sum(axis=0) * (len(frames_gray) - 1) / len(series)
    return math.exp(total[0]), float(total[1]), float(total[2])


def track_series(frames_gray: list[np.ndarray], mask: np.ndarray | None = None,
                 ransac_px: float = RANSAC_PX) -> np.ndarray | None:
    """Per frame pair (rows: pair index, dlog scale, centre dx, centre dy as shares of the frame) for the
    pairs that could be tracked, or None if too few could. Features are only picked where `mask` is non-zero."""
    h, w = frames_gray[0].shape
    centre = np.array([w / 2, h / 2])
    rows = []
    pts = _features(frames_gray[0], mask)
    for i, (a, b) in enumerate(zip(frames_gray, frames_gray[1:])):
        if pts is None or len(pts) < RESEED:
            pts = _features(a, mask)
        if pts is None or len(pts) < MIN_POINTS:
            pts = None
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
        keep = status.reshape(-1) == 1
        good_a, good_b = pts[keep], nxt[keep]
        if len(good_a) < MIN_POINTS:
            pts = None
            continue
        m, inliers = cv2.estimateAffinePartial2D(good_a, good_b, method=cv2.RANSAC, ransacReprojThreshold=ransac_px)
        if m is None or int(inliers.sum()) < MIN_POINTS:
            pts = None
            continue
        moved = m[:, :2] @ centre + m[:, 2]
        rows.append((i, math.log(math.hypot(m[0, 0], m[1, 0])), (moved[0] - centre[0]) / w, (moved[1] - centre[1]) / h))
        pts = good_b.reshape(-1, 1, 2)
    if len(rows) < max(1, (len(frames_gray) - 1) // 2):
        return None
    return np.array(rows, dtype=float)


def bound_motion(scale: float, dx: float, dy: float) -> tuple[float, float, float]:
    """Keep a camera move within what a re-render can reproduce without cropping to a tiny patch."""
    return (min(MAX_SCALE, max(1 / MAX_SCALE, scale)),
            min(MAX_DXY, max(-MAX_DXY, dx)), min(MAX_DXY, max(-MAX_DXY, dy)))


def extrapolate(scale: float, dx: float, dy: float, k: float) -> tuple[float, float, float]:
    """Stretch the motion measured over the analysed window by k (shot length / window length), bounded."""
    k = min(MAX_K, max(0.0, k))
    return bound_motion(scale ** k, dx * k, dy * k)


def classify(scale: float, dx: float, dy: float) -> dict:
    if scale >= 1 + ZOOM_MIN:
        kind = "zoom_in"
    elif scale <= 1 / (1 + ZOOM_MIN):
        kind = "zoom_out"
    elif math.hypot(dx, dy) >= PAN_MIN:
        kind = "pan"
    else:
        return dict(STATIC)
    return {"type": kind, "scale_from": 1.0, "scale_to": round(scale, 3) if kind != "pan" else 1.0,
            "dx": round(dx, 3), "dy": round(dy, 3)}


def _half(tin: dict) -> int:
    return 0 if tin["type"] == "cut" else tin["duration_ms"] // 2


def shot_window(shots: list[dict], i: int, cache_fps: int) -> tuple[int, int]:
    """First and last cached frame index to analyse for shot i (inclusive)."""
    s = shots[i]
    lo = s["start_ms"] + (_half(s["transition_in"]) if i else 0)
    hi = s["end_ms"] - (_half(shots[i + 1]["transition_in"]) if i + 1 < len(shots) else 0)
    first = math.ceil(lo * cache_fps / 1000) + EDGE_FRAMES
    last = math.floor(hi * cache_fps / 1000) - 1 - EDGE_FRAMES
    return first, last


def run(wd: WorkDir, source: Path) -> list[str]:
    if not wd.has(fr.OUTPUT):
        raise FileNotFoundError(f"{fr.OUTPUT} missing: run the frames stage first")
    cache = wd.read_json(fr.OUTPUT)
    shots = wd.read_json("shots.json")["shots"]
    x, y, w, h = fr.content_of(wd)        # track the picture, not the letterbox padding
    out, warnings = [], []
    for i, s in enumerate(shots):
        first, last = shot_window(shots, i, cache["fps"])
        last = min(last, len(cache["files"]) - 1)
        if last - first < 2:
            out.append(dict(STATIC))
            continue
        gray = [cv2.cvtColor(fr.cached_frame(wd, k)[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                for k in range(first, last + 1)]
        res = window_motion(gray)
        if res is None:
            warnings.append(f"shot {i + 1}: too few trackable points, marked static")
            out.append(dict(STATIC))
            continue
        scale, dx, dy = res
        k = (s["end_ms"] - s["start_ms"]) / ((last - first) * 1000 / cache["fps"])   # extrapolate to the shot
        out.append(classify(*extrapolate(scale, dx, dy, k)))
    wd.write_json(OUTPUT, {"shots": out})
    return warnings
