"""Frame cache (15 fps, 540x960 JPEGs) and a one-pass full-resolution crop reader."""
from pathlib import Path

import cv2
import numpy as np

from vtpl.util.ffmpeg import run_ffmpeg
from vtpl.util.workdir import WorkDir

OUTPUT = "frames.json"
CACHE_FPS, CACHE_W, CACHE_H = 15, 540, 960

Rect = tuple[int, int, int, int]       # x, y, w, h in pixels


def run(wd: WorkDir, source: Path) -> list[str]:
    d = wd.path("frames")
    d.mkdir(exist_ok=True)
    for old in d.glob("*.jpg"):
        old.unlink()
    run_ffmpeg(["-i", str(wd.path("norm.mp4")), "-vf", f"fps={CACHE_FPS},scale={CACHE_W}:{CACHE_H}",
                "-q:v", "3", "-start_number", "0", str(d / "%05d.jpg")])
    files = sorted(p.name for p in d.glob("*.jpg"))
    if not files:
        raise ValueError("no frames could be extracted from norm.mp4")
    wd.write_json(OUTPUT, {"fps": CACHE_FPS, "w": CACHE_W, "h": CACHE_H, "files": files})
    return []


def cache_time_ms(i: int, fps: int = CACHE_FPS) -> int:
    return round(i * 1000 / fps)


def cached_frame(wd: WorkDir, i: int) -> np.ndarray:
    path = wd.path("frames") / f"{i:05d}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cached frame missing: {path}")
    return img


def clamp_rect(r: Rect, fw: int, fh: int) -> Rect:
    x, y, w, h = r
    x0, y0 = min(max(0, x), fw), min(max(0, y), fh)
    x1, y1 = min(fw, max(0, x + w)), min(fh, max(0, y + h))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def read_crops(video: Path, requests: dict[int, list[tuple]]) -> dict[str, dict[int, np.ndarray]]:
    """Decode the video once and return {key: {frame_index: BGR crop}} for every requested crop.
    A request is (key, rect) or (key, rect, scale): a scale below 1 stores the crop downscaled."""
    out: dict[str, dict[int, np.ndarray]] = {}
    if not requests:
        return out
    last = max(requests)
    cap = cv2.VideoCapture(str(video))
    try:
        for i in range(last + 1):
            if i not in requests:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            fh, fw = frame.shape[:2]
            for key, rect, *scale in requests[i]:
                x, y, w, h = clamp_rect(rect, fw, fh)
                crop = frame[y:y + h, x:x + w]
                if scale and scale[0] != 1 and w and h:
                    size = (max(1, round(w * scale[0])), max(1, round(h * scale[0])))
                    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
                out.setdefault(key, {})[i] = crop.copy()
    finally:
        cap.release()
    return out
