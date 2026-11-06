"""Match a user clip's look to a template shot's grade with FFmpeg eq."""
from pathlib import Path

import cv2

from vtpl.analyzers.grade import SAMPLES, stats

LIMITS = {"brightness": (-0.3, 0.3), "contrast": (0.7, 1.5), "saturation": (0.5, 1.8)}
# eq works on limited-range (16..235) luma code values, pivoting contrast on code 128 and adding
# brightness*255 codes; grade statistics are measured on full-range decoded frames. In those units the
# contrast pivot is (128-16)/219 and a brightness b shifts the mean by b*255/219.
LIMITED = 219 / 255
PIVOT = (128 - 16) / 219


def _clamp(name: str, v: float) -> float:
    lo, hi = LIMITS[name]
    return round(min(hi, max(lo, v)), 3)


def clip_stats(path: Path, start_ms: int, dur_ms: int, n: int = SAMPLES) -> dict:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for k in range(n):
            cap.set(cv2.CAP_PROP_POS_MSEC, start_ms + dur_ms * k / max(1, n - 1))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"could not read frames from {path} to match its colour grade")
    return stats(frames)


def eq_params(target: dict, clip: dict) -> dict | None:
    """eq settings that move `clip` towards `target`, or None if no change is needed."""
    m, want = clip["luma_mean"], target["luma_mean"]
    c = _clamp("contrast", target["luma_std"] / clip["luma_std"]) if clip["luma_std"] > 1e-3 else 1.0
    b_raw = LIMITED * (want - ((m - PIVOT) * c + PIVOT))
    lo, hi = LIMITS["brightness"]
    if not lo <= b_raw <= hi and abs(m - PIVOT) > 1e-3:
        # brightness clips, and contrast (pivoting on mid-grey) would push the mean further off target:
        # the mean matters most, so take the contrast that reaches it with brightness at its limit
        b_lim = hi if b_raw > hi else lo
        c_mean = (want - PIVOT - b_lim / LIMITED) / (m - PIVOT)
        if c_mean > 0:
            c = _clamp("contrast", min(c_mean, c))
    b = _clamp("brightness", LIMITED * (want - ((m - PIVOT) * c + PIVOT)))
    s = _clamp("saturation", target["chroma"] / clip["chroma"]) if clip["chroma"] > 1e-3 else 1.0
    if abs(b) < 0.01 and abs(c - 1) < 0.02 and abs(s - 1) < 0.02:
        return None
    return {"brightness": b, "contrast": c, "saturation": s}


def eq_filter(p: dict) -> str:
    return f"eq=brightness={p['brightness']:.3f}:contrast={p['contrast']:.3f}:saturation={p['saturation']:.3f}"
