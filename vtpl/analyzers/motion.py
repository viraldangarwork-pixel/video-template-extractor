"""Per-shot camera motion (zoom / pan / static) from optical flow on cached frames."""
import math
from pathlib import Path

import cv2
import numpy as np

from vtpl import frames as fr
from vtpl.util.workdir import WorkDir

OUTPUT = "motion.json"
MIN_POINTS = 20
RESEED = 50
ZOOM_MIN = 0.04            # |scale - 1| for a zoom
PAN_MIN = 0.03             # centre movement, share of the frame, for a pan
EDGE_FRAMES = 1            # cached frames skipped at each end of a shot
STATIC = {"type": "static", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.0, "dy": 0.0}


def _features(gray: np.ndarray):
    return cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=8)


def track_motion(frames_gray: list[np.ndarray]) -> tuple[float, float, float] | None:
    """(total scale, centre dx, centre dy as shares of the frame), or None if too little can be tracked."""
    h, w = frames_gray[0].shape
    centre = np.array([w / 2, h / 2])
    scale, dx, dy, ok = 1.0, 0.0, 0.0, 0
    pts = _features(frames_gray[0])
    for a, b in zip(frames_gray, frames_gray[1:]):
        if pts is None or len(pts) < RESEED:
            pts = _features(a)
        if pts is None or len(pts) < MIN_POINTS:
            pts = None
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
        keep = status.reshape(-1) == 1
        good_a, good_b = pts[keep], nxt[keep]
        if len(good_a) < MIN_POINTS:
            pts = None
            continue
        m, _ = cv2.estimateAffinePartial2D(good_a, good_b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if m is None:
            pts = None
            continue
        moved = m[:, :2] @ centre + m[:, 2]
        scale *= math.hypot(m[0, 0], m[1, 0])
        dx += (moved[0] - centre[0]) / w
        dy += (moved[1] - centre[1]) / h
        ok += 1
        pts = good_b.reshape(-1, 1, 2)
    if ok < max(1, (len(frames_gray) - 1) // 2):
        return None
    return scale, dx, dy


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
    out, warnings = [], []
    for i, s in enumerate(shots):
        first, last = shot_window(shots, i, cache["fps"])
        last = min(last, len(cache["files"]) - 1)
        if last - first < 2:
            out.append(dict(STATIC))
            continue
        gray = [cv2.cvtColor(fr.cached_frame(wd, k), cv2.COLOR_BGR2GRAY) for k in range(first, last + 1)]
        res = track_motion(gray)
        if res is None:
            warnings.append(f"shot {i + 1}: too few trackable points, marked static")
            out.append(dict(STATIC))
            continue
        scale, dx, dy = res
        k = (s["end_ms"] - s["start_ms"]) / ((last - first) * 1000 / cache["fps"])   # extrapolate to the shot
        out.append(classify(scale ** k, dx * k, dy * k))
    wd.write_json(OUTPUT, {"shots": out})
    return warnings
