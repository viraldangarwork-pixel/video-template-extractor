"""Synthetic text-overlay media drawn with Pillow, so analyser tests don't depend on vtpl's ASS renderer."""
import math
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vtpl.fontlib import entry, font_path

W, H, FPS = 1080, 1920, 30


@dataclass(frozen=True)
class TextItem:
    text: str
    font_id: str
    size: int                                   # Pillow em size, px
    color: tuple[int, int, int]                 # RGB
    center: tuple[int, int]                     # px
    start_ms: int
    end_ms: int
    stroke: tuple[tuple[int, int, int], int] | None = None    # (RGB, px)
    shadow: bool = False
    anim_in: str = "none"                       # none | pop | fade | slide | typewriter
    anim_out: str = "none"                      # none | pop | fade
    anim_ms: int = 200
    slide_dx: int = -300                        # px offset at the start of a slide-in


BASIC_ITEMS = [
    TextItem("WAIT FOR IT", "Montserrat-Bold", 96, (255, 255, 255), (540, 1300), 500, 3000,
             stroke=((0, 0, 0), 6), anim_in="pop", anim_out="fade"),
    TextItem("DAY 1", "Anton-Regular", 110, (255, 214, 0), (540, 500), 1000, 3500,
             shadow=True, anim_in="slide"),
    TextItem("SALE", "BebasNeue-Regular", 160, (230, 30, 60), (540, 900), 1500, 3800,
             stroke=((255, 255, 255), 4), anim_in="typewriter", anim_ms=400),
]


# On the moving background: a hard cut, a typewriter reveal and a text shown from the first frame
# to past the end of the video. MOVING_SECONDS long.
MOVING_SECONDS = 4.0
MOVING_ITEMS = [
    TextItem("HARD CUT", "Anton-Regular", 120, (255, 255, 255), (540, 700), 800, 2600, stroke=((0, 0, 0), 5)),
    TextItem("TYPE IT OUT", "Montserrat-Bold", 90, (255, 214, 0), (540, 1100), 1500, 3500,
             anim_in="typewriter", anim_ms=400),
    TextItem("ALWAYS ON", "BebasNeue-Regular", 130, (255, 255, 255), (540, 1500), 0, 10_000, stroke=((0, 0, 0), 5)),
]


@lru_cache(maxsize=256)
def _font(font_id: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(entry(font_id))), size)


def text_layer(item: TextItem, scale: float = 1.0, chars: int | None = None) -> Image.Image:
    """RGBA image of the text with stroke/shadow. Laid out for the full string, so partial
    (typewriter) text stays left-aligned where the full text will be."""
    font = _font(item.font_id, max(4, round(item.size * scale)))
    sw = round(item.stroke[1] * scale) if item.stroke else 0
    x0, y0, x1, y1 = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), item.text, font=font, stroke_width=sw)
    pad = 12
    layer = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    org = (pad - x0, pad - y0)
    shown = item.text if chars is None else item.text[:chars]
    if item.shadow:
        off = max(1, round(5 * scale))
        draw.text((org[0] + off, org[1] + off), shown, font=font, fill=(0, 0, 0, 170),
                  stroke_width=sw, stroke_fill=(0, 0, 0, 170))
    draw.text(org, shown, font=font, fill=(*item.color, 255), stroke_width=sw,
              stroke_fill=(*item.stroke[0], 255) if item.stroke else None)
    return layer


def _composite(base: Image.Image, layer: Image.Image, cx: int, cy: int) -> None:
    x, y = cx - layer.width // 2, cy - layer.height // 2
    sx, sy = max(0, -x), max(0, -y)
    ex, ey = min(layer.width, base.width - x), min(layer.height, base.height - y)
    if ex > sx and ey > sy:
        base.alpha_composite(layer.crop((sx, sy, ex, ey)), dest=(x + sx, y + sy))


def _pop(p: float) -> float:
    """0.8 -> 1.15 over the first two thirds, then back to 1.0."""
    return 0.8 + 0.35 * p / (2 / 3) if p < 2 / 3 else 1.15 - 0.15 * (p - 2 / 3) / (1 / 3)


def _state(item: TextItem, t: int):
    """(scale, alpha, dx, chars) at time t (ms), or None while hidden."""
    if not item.start_ms <= t < item.end_ms:
        return None
    p_in = (t - item.start_ms) / item.anim_ms
    p_out = (item.end_ms - t) / item.anim_ms
    scale, alpha, dx, chars = 1.0, 1.0, 0, None
    if p_in < 1:
        if item.anim_in == "pop":
            scale = _pop(p_in)
        elif item.anim_in == "fade":
            alpha = p_in
        elif item.anim_in == "slide":
            dx = round(item.slide_dx * (1 - p_in))
        elif item.anim_in == "typewriter":
            chars = max(1, math.ceil(p_in * len(item.text)))
    if p_out < 1:
        if item.anim_out == "pop":
            scale *= max(0.05, p_out)
        elif item.anim_out == "fade":
            alpha *= p_out
    return scale, alpha, dx, chars


def background(kind: str, t_ms: int) -> np.ndarray:
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    if kind == "moving":
        y = (y + t_ms / 8000.0) % 1.0
    top, bottom = np.array([30, 60, 120], np.float32), np.array([110, 40, 90], np.float32)
    return np.broadcast_to(top + (bottom - top) * y, (H, W, 3)).astype(np.uint8)


def frame_at(items: list[TextItem], t_ms: int, bg: str = "gradient") -> np.ndarray:
    """One RGB frame at time t_ms."""
    base = Image.fromarray(background(bg, t_ms)).convert("RGBA")
    for it in items:
        st = _state(it, t_ms)
        if st is None:
            continue
        scale, alpha, dx, chars = st
        layer = text_layer(it, scale, chars)
        if alpha < 1:
            layer.putalpha(layer.getchannel("A").point(lambda v: round(v * alpha)))
        _composite(base, layer, it.center[0] + dx, it.center[1])
    return np.array(base.convert("RGB"))


def truth(it: TextItem) -> dict:
    return {"text": it.text, "font": it.font_id, "size": it.size, "color": "#%02X%02X%02X" % it.color,
            "stroke": it.stroke is not None, "shadow": it.shadow, "center": (it.center[0] / W, it.center[1] / H),
            "start_ms": it.start_ms, "end_ms": it.end_ms, "anim_in": it.anim_in, "anim_out": it.anim_out}


def text_video(path: Path, items: list[TextItem], seconds: float, bg: str = "gradient") -> list[dict]:
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for i in range(round(seconds * FPS)):
        proc.stdin.write(frame_at(items, round(i * 1000 / FPS), bg).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    return [truth(it) for it in items]


def text_crop(text: str, font_id: str, size: int, color: tuple[int, int, int], stroke=None, shadow=False,
              bg_rgb: tuple[int, int, int] = (40, 80, 140), pad: int = 40) -> np.ndarray:
    """A BGR crop of one text on a solid background, like a box cut from a video frame."""
    layer = text_layer(TextItem(text, font_id, size, color, (0, 0), 0, 1, stroke, shadow))
    base = Image.new("RGBA", (layer.width + 2 * pad, layer.height + 2 * pad), (*bg_rgb, 255))
    base.alpha_composite(layer, dest=(pad, pad))
    return np.array(base.convert("RGB"))[:, :, ::-1].copy()
