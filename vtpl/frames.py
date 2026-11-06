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


CONTENT_INSET = 2         # cache pixels kept clear of the padding edge when measuring the picture


def cached_content(meta: dict) -> Rect:
    """The source picture's area (meta.json `content`, without the black padding) in cache pixels.
    A meta.json written before `content` existed means the whole frame."""
    c = meta.get("content")
    if not c:
        return 0, 0, CACHE_W, CACHE_H
    sx, sy = CACHE_W / meta["w"], CACHE_H / meta["h"]
    x0, y0 = round(c["x"] * sx), round(c["y"] * sy)
    x1, y1 = round((c["x"] + c["w"]) * sx), round((c["y"] + c["h"]) * sy)
    if x0 > 0 or x1 < CACHE_W:
        x0, x1 = x0 + CONTENT_INSET, x1 - CONTENT_INSET
    if y0 > 0 or y1 < CACHE_H:
        y0, y1 = y0 + CONTENT_INSET, y1 - CONTENT_INSET
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0, 0, CACHE_W, CACHE_H
    return x0, y0, x1 - x0, y1 - y0


def content_of(wd: WorkDir) -> Rect:
    return cached_content(wd.read_json("meta.json")) if wd.has("meta.json") else (0, 0, CACHE_W, CACHE_H)


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
