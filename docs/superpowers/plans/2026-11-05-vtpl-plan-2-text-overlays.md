# vtpl Plan 2 — On-screen Text and Captions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vtpl analyze` also extracts on-screen text overlays (text, timing, position, colour, stroke, shadow, font, size, in/out animation) and tells captions apart from overlays; `vtpl render` burns overlays and karaoke captions back in with libass, using the user's replacement text.

**Architecture:** Two new stages: `frames` (15 fps 540×960 JPEG cache) and `text` (OCR samples → tracks → per-track refinement from one full-resolution decode pass). Each step of the text stage is a pure module under `vtpl/analyzers/text/` with its own tests. `merge` adds text items to the template and moves speech-matching text into captions. The renderer builds one ASS file and adds a `subtitles` filter to the existing FFmpeg graph. A bundled font set (Google Fonts via Fontsource) is used for font guessing, fixtures and rendering.

**Tech Stack:** Plan 1 stack plus RapidOCR (`rapidocr-onnxruntime`, models bundled in the wheel), `rapidfuzz`, Pillow, OpenCV k-means/distance transforms, libass via FFmpeg's `subtitles` filter.

**Spec:** `docs/superpowers/specs/2026-11-05-video-template-extractor-design.md` §6.5, §6.8, §7.4–7.5.

**Branch:** `viral/feature/vtpl-text-overlays` (created from `viral/feature/vtpl-core-pipeline`).

**Environment facts verified while writing this plan:**
- `rapidocr-onnxruntime` 1.2.3 installs on Python 3.13; `RapidOCR()(img_bgr)` returns `([[quad, text, score_str], ...], elapsed)` — **the score is a string**, convert with `float()`. It reads 540-wide text in ~0.1–0.7 s per frame on CPU.
- Static per-weight TTFs download from `https://cdn.jsdelivr.net/fontsource/fonts/<slug>@latest/latin-<weight>-normal.ttf` (35–50 KB each).
- FFmpeg 6.1 here has the `subtitles` (libass) filter.

---

## File structure (Plan 2)

| File | Responsibility |
|---|---|
| `scripts/fetch_fonts.py` | One-off: download bundled fonts, write `catalog.json` |
| `vtpl/assets/fonts/*.ttf`, `catalog.json`, `README.md` | Bundled OFL/Apache fonts + licence note |
| `vtpl/fontlib.py` | Font catalog, paths, Pillow fonts, ASS font-size conversion |
| `vtpl/frames.py` | `frames` stage (JPEG cache), cached-frame access, one-pass full-res crop reader |
| `vtpl/analyzers/text/__init__.py` | Package marker |
| `vtpl/analyzers/text/detect.py` | RapidOCR wrapper, OCR every 2nd cached frame |
| `vtpl/analyzers/text/track.py` | Link detections into tracks (pure) |
| `vtpl/analyzers/text/refine.py` | Exact first/last frame of a track (pure) |
| `vtpl/analyzers/text/style.py` | Fill colour, stroke, shadow, glyph mask from a crop (pure) |
| `vtpl/analyzers/text/font.py` | Font ranking by rendering candidates; em size (pure) |
| `vtpl/analyzers/text/anim.py` | In/out animation measurement and classification (pure) |
| `vtpl/analyzers/text/stage.py` | `text` stage orchestration → `text.json` |
| `vtpl/merge.py`, `vtpl/schema.py`, `vtpl/pipeline.py` | Text items, caption dedupe, schema, stage registration |
| `vtpl/render/ass.py` | Build the ASS document (pure) |
| `vtpl/render/graph.py`, `vtpl/render/run.py`, `vtpl/render/slots.py` | Burn ASS into the render; `captions` replacement key |
| `tests/synth/text_fixtures.py` | Pillow-drawn text videos/crops with ground truth (independent of libass) |

---

### Task 1: Bundled fonts and font library

**Files:**
- Create: `scripts/fetch_fonts.py`, `vtpl/assets/fonts/README.md`, `vtpl/fontlib.py`, generated `vtpl/assets/fonts/catalog.json` + `*.ttf`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_fontlib.py`

- [ ] **Step 1: Add dependencies and package data to `pyproject.toml`**

In `[project] dependencies` add:
```toml
  "rapidocr-onnxruntime>=1.2.3",
  "rapidfuzz>=3.0",
  "Pillow>=10.0",
```
And add a new table:
```toml
[tool.setuptools.package-data]
vtpl = ["assets/fonts/*.ttf", "assets/fonts/*.json", "assets/fonts/*.md"]
```
Run: `/home/user/.venv/bin/pip install -e '.[dev]'` — Expected: `Successfully installed ... vtpl-0.1.0` (rapidocr/rapidfuzz may already be present).

- [ ] **Step 2: Write `scripts/fetch_fonts.py`**

```python
"""Download the bundled font set from Fontsource (Google Fonts; OFL or Apache-2.0) and write catalog.json.

Run once from the repo root:  python scripts/fetch_fonts.py
Then commit vtpl/assets/fonts/. The family name in catalog.json is read from each file, so it always
matches what libass and Pillow see.
"""
import json
import sys
import urllib.request
from pathlib import Path

from PIL import ImageFont

DEST = Path(__file__).resolve().parents[1] / "vtpl" / "assets" / "fonts"
URL = "https://cdn.jsdelivr.net/fontsource/fonts/{slug}@latest/latin-{weight}-normal.ttf"

FONTS = [  # (id, fontsource slug, weight) — at most one regular (<600) and one bold (>=600) per family
    ("Montserrat-Regular", "montserrat", 400), ("Montserrat-Bold", "montserrat", 700),
    ("Poppins-Regular", "poppins", 400), ("Poppins-Bold", "poppins", 700),
    ("Roboto-Regular", "roboto", 400), ("Roboto-Bold", "roboto", 700),
    ("BebasNeue-Regular", "bebas-neue", 400), ("Anton-Regular", "anton", 400),
    ("Oswald-Bold", "oswald", 700), ("OpenSans-Bold", "open-sans", 700),
    ("Lato-Bold", "lato", 700), ("Raleway-Bold", "raleway", 700),
    ("LeagueSpartan-Bold", "league-spartan", 700), ("DMSans-Bold", "dm-sans", 700),
    ("Nunito-Bold", "nunito", 700), ("Rubik-Bold", "rubik", 700),
    ("WorkSans-Bold", "work-sans", 700), ("Inter-Bold", "inter", 700),
    ("ArchivoBlack-Regular", "archivo-black", 400), ("Bangers-Regular", "bangers", 400),
    ("LuckiestGuy-Regular", "luckiest-guy", 400), ("PermanentMarker-Regular", "permanent-marker", 400),
    ("Pacifico-Regular", "pacifico", 400), ("Lobster-Regular", "lobster", 400),
    ("DancingScript-Bold", "dancing-script", 700), ("Caveat-Bold", "caveat", 700),
    ("PlayfairDisplay-Bold", "playfair-display", 700), ("Merriweather-Bold", "merriweather", 700),
    ("Lora-Bold", "lora", 700), ("AbrilFatface-Regular", "abril-fatface", 400),
    ("Righteous-Regular", "righteous", 400), ("Fredoka-Bold", "fredoka", 700),
    ("Kanit-Bold", "kanit", 700), ("BarlowCondensed-Bold", "barlow-condensed", 700),
    ("Teko-Bold", "teko", 700), ("AlfaSlabOne-Regular", "alfa-slab-one", 400),
    ("BlackOpsOne-Regular", "black-ops-one", 400), ("Bungee-Regular", "bungee", 400),
    ("Staatliches-Regular", "staatliches", 400), ("TitanOne-Regular", "titan-one", 400),
    ("PassionOne-Regular", "passion-one", 400), ("FjallaOne-Regular", "fjalla-one", 400),
]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    catalog, failed = [], []
    for font_id, slug, weight in FONTS:
        file = f"{slug}-{weight}.ttf"
        path = DEST / file
        try:
            if not path.exists():
                urllib.request.urlretrieve(URL.format(slug=slug, weight=weight), path)
            family = ImageFont.truetype(str(path), 40).getname()[0]
        except Exception as e:
            failed.append(f"{font_id}: {e}")
            path.unlink(missing_ok=True)
            continue
        catalog.append({"id": font_id, "family": family, "weight": weight, "file": file, "slug": slug})
    (DEST / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"{len(catalog)} fonts written to {DEST}")
    if failed:
        print("FAILED (remove these from FONTS and re-run):\n" + "\n".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run it**

Run: `/home/user/.venv/bin/python scripts/fetch_fonts.py`
Expected: `42 fonts written to .../vtpl/assets/fonts` and exit 0. If TLS fails behind a proxy, re-run with `SSL_CERT_FILE` pointing at the proxy CA bundle (here: `SSL_CERT_FILE=/root/.ccr/ca-bundle.crt`). If a font 404s, delete its line from `FONTS` and re-run; keep at least 30 fonts and keep `Montserrat-Bold`, `BebasNeue-Regular`, `Anton-Regular`, `Pacifico-Regular`, `PlayfairDisplay-Bold` (tests use them).

- [ ] **Step 4: Write `vtpl/assets/fonts/README.md`**

```markdown
# Bundled fonts

Static Latin subsets of Google Fonts, downloaded from Fontsource by `scripts/fetch_fonts.py`.
All are licensed under the SIL Open Font License 1.1 or the Apache License 2.0, which allow
redistribution. `catalog.json` lists each font's id, family name (read from the file), weight and file.
```

- [ ] **Step 5: Write the failing test** — `tests/unit/test_fontlib.py`

```python
import pytest
from PIL import ImageFont

from vtpl import fontlib


def test_catalog_files_match_family_names():
    cat = fontlib.catalog()
    assert len(cat) >= 30
    ids = [e.id for e in cat]
    assert len(ids) == len(set(ids)) and fontlib.FALLBACK_ID in ids
    for e in cat:
        assert ImageFont.truetype(str(fontlib.font_path(e)), 40).getname()[0] == e.family


def test_at_most_one_regular_and_one_bold_per_family():
    seen = set()
    for e in fontlib.catalog():
        assert (e.family, e.bold) not in seen
        seen.add((e.family, e.bold))


def test_entry_lookup():
    e = fontlib.entry("Montserrat-Bold")
    assert e.family == "Montserrat" and e.bold and e.weight == 700
    with pytest.raises(KeyError, match="unknown font 'Nope'"):
        fontlib.entry("Nope")


def test_ass_font_size_is_line_height():
    asc, desc = fontlib.pil_font("Montserrat-Bold", 100).getmetrics()
    assert fontlib.ass_font_size("Montserrat-Bold", 100) == asc + desc
    assert fontlib.ass_font_size("Montserrat-Bold", 100) > 100


def test_fonts_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VTPL_FONTS_DIR", str(tmp_path))
    assert fontlib.fonts_dir() == tmp_path
```

- [ ] **Step 6: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_fontlib.py -v`
Expected: FAIL with `ImportError: cannot import name 'fontlib' from 'vtpl'`

- [ ] **Step 7: Implement** — `vtpl/fontlib.py`

```python
"""Bundled font catalog: lookup, Pillow fonts and ASS size conversion."""
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

DEFAULT_DIR = Path(__file__).parent / "assets" / "fonts"
FALLBACK_ID = "Montserrat-Bold"


@dataclass(frozen=True)
class FontEntry:
    id: str
    family: str
    weight: int
    file: str
    slug: str

    @property
    def bold(self) -> bool:
        return self.weight >= 600


def fonts_dir() -> Path:
    return Path(os.environ.get("VTPL_FONTS_DIR", DEFAULT_DIR))


@lru_cache(maxsize=8)
def _load(directory: str) -> tuple[FontEntry, ...]:
    data = json.loads((Path(directory) / "catalog.json").read_text())
    return tuple(FontEntry(**e) for e in data)


def catalog() -> list[FontEntry]:
    return list(_load(str(fonts_dir())))


def entry(font_id: str) -> FontEntry:
    for e in catalog():
        if e.id == font_id:
            return e
    raise KeyError(f"unknown font {font_id!r}; see {fonts_dir() / 'catalog.json'}")


def font_path(e: FontEntry) -> Path:
    return fonts_dir() / e.file


@lru_cache(maxsize=1024)
def pil_font(font_id: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(entry(font_id))), size)


def ass_font_size(font_id: str, em_px: int) -> int:
    """libass sizes fonts by line height (ascent + descent), not by em, so convert."""
    ascent, descent = pil_font(font_id, em_px).getmetrics()
    return ascent + descent
```

- [ ] **Step 8: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_fontlib.py -v`
Expected: 5 passed

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml scripts/fetch_fonts.py vtpl/assets vtpl/fontlib.py tests/unit/test_fontlib.py
git commit -m "feat: bundled font set and font library"
```

---

### Task 2: Text fixture generator (Pillow, independent of libass)

**Files:**
- Create: `tests/synth/text_fixtures.py`
- Modify: `tests/conftest.py` (add `text_video_basic`)
- Test: `tests/unit/test_text_fixtures.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_text_fixtures.py`

```python
import numpy as np
import pytest

from tests.synth.text_fixtures import BASIC_ITEMS, TextItem, frame_at, text_crop
from vtpl.util.ffmpeg import duration_ms


def _white_pixels(rgb, cx, cy, half_w=250, half_h=60):
    region = rgb[cy - half_h:cy + half_h, cx - half_w:cx + half_w]
    return int((region.min(axis=2) > 230).sum())


def test_frame_at_shows_items_only_while_visible():
    item = BASIC_ITEMS[0]            # "WAIT FOR IT", 500-3000 ms, white
    assert _white_pixels(frame_at(BASIC_ITEMS, 200), *item.center) == 0
    assert _white_pixels(frame_at(BASIC_ITEMS, 2000), *item.center) > 500


def test_fade_in_is_partial_midway():
    it = TextItem("HELLO", "Montserrat-Bold", 120, (255, 255, 255), (540, 960), 0, 2000, anim_in="fade", anim_ms=400)
    mid, done = frame_at([it], 200), frame_at([it], 600)
    assert _white_pixels(mid, 540, 960) < _white_pixels(done, 540, 960) // 4


def test_typewriter_reveals_left_part_first():
    it = TextItem("SALE", "BebasNeue-Regular", 200, (255, 255, 255), (540, 960), 0, 2000,
                  anim_in="typewriter", anim_ms=400)
    early = frame_at([it], 50)
    cols = np.nonzero((early[860:1060].min(axis=2) > 230).any(axis=0))[0]
    assert len(cols) and cols.max() < 540         # only the first letter, left of centre


def test_text_crop_is_bgr_with_background_border():
    crop = text_crop("HI", "Montserrat-Bold", 80, (255, 255, 255), bg_rgb=(10, 20, 200))
    assert crop.shape[2] == 3 and tuple(crop[0, 0]) == (200, 20, 10)


@pytest.mark.slow
def test_basic_video(text_video_basic):
    assert abs(duration_ms(text_video_basic["path"]) - 4000) <= 50
    assert [t["text"] for t in text_video_basic["truth"]] == ["WAIT FOR IT", "DAY 1", "SALE"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.synth.text_fixtures'`

- [ ] **Step 3: Implement** — `tests/synth/text_fixtures.py`

```python
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
```

Append to `tests/conftest.py`:
```python
@pytest.fixture(scope="session")
def text_video_basic(fixtures_dir):
    _need_ffmpeg()
    from tests.synth.text_fixtures import BASIC_ITEMS, text_video
    path = fixtures_dir / "text_basic.mp4"
    return {"path": path, "truth": text_video(path, BASIC_ITEMS, 4.0)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_fixtures.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tests/synth/text_fixtures.py tests/conftest.py tests/unit/test_text_fixtures.py
git commit -m "test: Pillow-drawn text overlay fixtures with ground truth"
```

---

### Task 3: Frame cache stage and one-pass crop reader

**Files:**
- Create: `vtpl/frames.py`
- Test: `tests/unit/test_frames.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from vtpl import frames, ingest
from vtpl.util.workdir import WorkDir


def test_clamp_rect():
    assert frames.clamp_rect((-10, -10, 50, 50), 100, 100) == (0, 0, 40, 40)
    assert frames.clamp_rect((90, 95, 50, 50), 100, 100) == (90, 95, 10, 5)
    assert frames.clamp_rect((200, 0, 10, 10), 100, 100) == (100, 0, 0, 10)


def test_cache_time_ms():
    assert frames.cache_time_ms(30) == 2000 and frames.cache_time_ms(1) == 67


@pytest.mark.slow
def test_frames_stage(tmp_path, small_cards_video):
    wd = WorkDir(tmp_path / "w")
    ingest.run(wd, small_cards_video)
    assert frames.run(wd, small_cards_video) == []
    doc = wd.read_json("frames.json")
    assert doc["fps"] == 15 and (doc["w"], doc["h"]) == (540, 960)
    assert 88 <= len(doc["files"]) <= 92                      # 6 s at 15 fps
    first = frames.cached_frame(wd, 0)
    assert first.shape == (960, 540, 3) and first[..., 2].mean() > 200     # red card, BGR


@pytest.mark.slow
def test_read_crops_single_pass(cards_video):
    req = {0: [("a", (0, 0, 100, 100))],
           50: [("a", (10, 10, 50, 50)), ("b", (-20, -20, 60, 60))],
           130: [("c", (1000, 1800, 500, 500))]}
    out = frames.read_crops(cards_video["path"], req)
    assert out["a"][0].shape == (100, 100, 3) and out["a"][0][..., 2].mean() > 200   # red
    assert out["a"][50][..., 0].mean() > 200                                          # blue
    assert out["b"][50].shape == (40, 40, 3)
    assert out["c"][130].shape == (120, 80, 3)                                        # clamped at frame edge
    assert frames.read_crops(cards_video["path"], {}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_frames.py -v`
Expected: FAIL with `ImportError: cannot import name 'frames' from 'vtpl'`

- [ ] **Step 3: Implement** — `vtpl/frames.py`

```python
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


def read_crops(video: Path, requests: dict[int, list[tuple[str, Rect]]]) -> dict[str, dict[int, np.ndarray]]:
    """Decode the video once and return {key: {frame_index: BGR crop}} for every requested crop."""
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
            for key, rect in requests[i]:
                x, y, w, h = clamp_rect(rect, fw, fh)
                out.setdefault(key, {})[i] = frame[y:y + h, x:x + w].copy()
    finally:
        cap.release()
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_frames.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/frames.py tests/unit/test_frames.py
git commit -m "feat: frame cache stage and one-pass crop reader"
```

---

### Task 4: OCR detection

**Files:**
- Create: `vtpl/analyzers/text/__init__.py` (empty), `vtpl/analyzers/text/detect.py`
- Test: `tests/unit/test_text_detect.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from rapidfuzz import fuzz

from tests.synth.text_fixtures import text_crop
from vtpl import frames, ingest
from vtpl.analyzers.text import detect
from vtpl.util.workdir import WorkDir


def test_ocr_reads_rendered_text():
    img = text_crop("WAIT FOR IT", "Montserrat-Bold", 64, (255, 255, 255), stroke=((0, 0, 0), 4))
    dets = detect.ocr(img)
    best = max(dets, key=lambda d: d["score"])
    assert fuzz.ratio(best["text"].upper(), "WAIT FOR IT") >= 85 and best["score"] >= 0.6
    x, y, w, h = best["box"]
    assert x < 80 and w > img.shape[1] * 0.6 and all(isinstance(v, int) for v in best["box"])


def test_to_norm():
    assert detect.to_norm([54, 96, 270, 48], 540, 960) == [0.1, 0.1, 0.5, 0.05]


@pytest.mark.slow
def test_detect_samples_on_fixture(tmp_path, text_video_basic):
    wd = WorkDir(tmp_path / "w")
    ingest.run(wd, text_video_basic["path"])
    frames.run(wd, text_video_basic["path"])
    samples = detect.detect_samples(wd)
    assert [s["t_ms"] for s in samples[:3]] == [0, 133, 267]
    assert [s["frame"] for s in samples[:3]] == [0, 4, 8]
    late = next(s for s in samples if s["t_ms"] >= 2400)
    texts = [d["text"].upper() for d in late["dets"]]
    for want in ["WAIT FOR IT", "DAY 1", "SALE"]:
        assert any(fuzz.ratio(t, want) >= 80 for t in texts), (want, texts)
    for d in late["dets"]:
        assert all(0 <= v <= 1 for v in d["box"]) and d["score"] >= detect.MIN_SCORE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.text'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/text/detect.py`

```python
"""OCR on every SAMPLE_EVERY-th cached frame (RapidOCR, CPU)."""
from functools import lru_cache

import numpy as np

from vtpl import frames as fr
from vtpl.util.workdir import WorkDir

SAMPLE_EVERY = 2          # every 2nd cached frame: one sample per ~133 ms at 15 fps
MIN_SCORE = 0.6
SMALL_BOX_PX = 20         # boxes shorter than this in the 540x960 cache are re-read at full resolution
SMALL_MIN_SCORE = 0.3     # keep weak small detections until the full-resolution re-read


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr(img_bgr: np.ndarray) -> list[dict]:
    """[{box: [x, y, w, h] int px, text, score}] for one BGR image."""
    result, _ = _engine()(img_bgr)
    out = []
    for quad, text, score in result or []:
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        x0, y0 = min(xs), min(ys)
        text = text.strip()
        if text:
            out.append({"box": [round(x0), round(y0), round(max(xs) - x0), round(max(ys) - y0)],
                        "text": text, "score": float(score)})
    return out


def to_norm(box_px: list[int], w: int, h: int) -> list[float]:
    return [box_px[0] / w, box_px[1] / h, box_px[2] / w, box_px[3] / h]


def detect_samples(wd: WorkDir) -> list[dict]:
    """[{t_ms, frame (full-rate index), dets: [{box (normalised), text, score}]}] in time order."""
    cache = wd.read_json(fr.OUTPUT)
    ratio = round(wd.read_json("meta.json")["fps"] / cache["fps"])
    samples, small = [], []
    for ci in range(0, len(cache["files"]), SAMPLE_EVERY):
        dets = []
        for d in ocr(fr.cached_frame(wd, ci)):
            is_small = d["box"][3] < SMALL_BOX_PX
            if d["score"] < (SMALL_MIN_SCORE if is_small else MIN_SCORE):
                continue
            dets.append({"box": to_norm(d["box"], cache["w"], cache["h"]), "text": d["text"],
                         "score": round(d["score"], 3)})
            if is_small:
                small.append((len(samples), len(dets) - 1))
        samples.append({"t_ms": fr.cache_time_ms(ci, cache["fps"]), "frame": ci * ratio, "dets": dets})
    _reread_small(wd, samples, small)
    for s in samples:
        s["dets"] = [d for d in s["dets"] if d["score"] >= MIN_SCORE]
    return samples


def _reread_small(wd: WorkDir, samples: list[dict], small: list[tuple[int, int]]) -> None:
    if not small:
        return
    meta = wd.read_json("meta.json")
    requests: dict[int, list] = {}
    for si, di in small:
        x, y, w, h = samples[si]["dets"][di]["box"]
        px = (round(x * meta["w"]), round(y * meta["h"]), round(w * meta["w"]), round(h * meta["h"]))
        pad = px[3] // 2 + 4
        rect = (px[0] - pad, px[1] - pad, px[2] + 2 * pad, px[3] + 2 * pad)
        requests.setdefault(samples[si]["frame"], []).append((f"{si}:{di}", rect))
    crops = fr.read_crops(wd.path("norm.mp4"), requests)
    for si, di in small:
        frame = samples[si]["frame"]
        crop = crops.get(f"{si}:{di}", {}).get(frame)
        if crop is None or crop.size == 0:
            continue
        found = ocr(crop)
        if found:
            best = max(found, key=lambda d: d["score"])
            det = samples[si]["dets"][di]
            if best["score"] > det["score"]:
                det["text"], det["score"] = best["text"], round(best["score"], 3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_detect.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/text tests/unit/test_text_detect.py
git commit -m "feat: OCR text detection on sampled cached frames"
```

---

### Task 5: Text tracking (pure)

**Files:**
- Create: `vtpl/analyzers/text/track.py`
- Test: `tests/unit/test_text_track.py`

Background: a *track* is one on-screen text over time. Detections link to a track when boxes overlap (IoU ≥ 0.5) and texts are ≥ 80 % similar — or when one text is a prefix of the other on the same line (typewriter reveal, where the box grows). Up to 3 missed samples are tolerated.

- [ ] **Step 1: Write the failing test**

```python
from vtpl.analyzers.text.track import build_tracks, iou, is_prefix

STEP = 133


def det(text, box, score=0.9):
    return {"text": text, "box": list(box), "score": score}


def samples(rows):
    """rows: list of det lists, one per sample."""
    return [{"t_ms": i * STEP, "frame": i * 4, "dets": d} for i, d in enumerate(rows)]


A = (0.1, 0.6, 0.8, 0.05)
B = (0.3, 0.2, 0.4, 0.06)


def test_iou_and_prefix():
    assert iou(A, A) == 1.0 and iou(A, B) == 0.0
    assert is_prefix("sa", "SALE") and is_prefix("SALE", "s a") is False and not is_prefix("", "SALE")


def test_two_texts_two_tracks_sorted_by_start():
    s = samples([[det("WAIT", A)], [det("WAIT", A), det("DAY 1", B)], [det("WAIT", A), det("DAY 1", B)],
                 [det("WAIT", A), det("DAY 1", B)], [det("DAY 1", B)]])
    tr = build_tracks(s, STEP)
    assert [(t["id"], t["text"], t["start_ms"], t["end_ms"]) for t in tr] == [
        ("t1", "WAIT", 0, 3 * STEP), ("t2", "DAY 1", STEP, 4 * STEP)]
    assert [o["frame"] for o in tr[0]["obs"]] == [0, 4, 8, 12]


def test_gap_of_three_samples_bridged_four_splits():
    rows = [[det("HELLO", A)]] * 2 + [[]] * 3 + [[det("HELLO", A)]] * 2
    assert len(build_tracks(samples(rows), STEP)) == 1
    rows = [[det("HELLO", A)]] * 3 + [[]] * 4 + [[det("HELLO", A)]] * 3
    assert len(build_tracks(samples(rows), STEP)) == 2


def test_typewriter_prefix_growth_is_one_track_with_full_text():
    boxes = [(0.40, 0.4, 0.05, 0.08), (0.40, 0.4, 0.10, 0.08), (0.40, 0.4, 0.16, 0.08), (0.40, 0.4, 0.2, 0.08)]
    rows = [[det(t, b)] for t, b in zip(["S", "SA", "SAL", "SALE"], boxes)] + [[det("SALE", boxes[-1])]] * 3
    tr = build_tracks(samples(rows), STEP)
    assert len(tr) == 1 and tr[0]["text"] == "SALE" and tr[0]["box"] == list(boxes[-1])


def test_ocr_noise_votes_and_short_blips_dropped():
    rows = [[det("WAIT FOR IT", A)], [det("WAIT F0R IT", A)], [det("WAIT FOR IT", A)], [det("X", B)]]
    tr = build_tracks(samples(rows), STEP)
    assert [t["text"] for t in tr] == ["WAIT FOR IT"]          # "X" lasts one sample (< 300 ms)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_track.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.text.track'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/text/track.py`

```python
"""Link per-sample OCR detections into text tracks. Boxes are normalised [x, y, w, h]."""
from collections import Counter

import numpy as np
from rapidfuzz import fuzz

IOU_MIN = 0.5
TEXT_MIN = 80
MAX_GAP = 3              # missed samples tolerated inside a track
MIN_TRACK_MS = 300


def iou(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _y_iou(a, b) -> float:
    inter = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = max(a[1] + a[3], b[1] + b[3]) - min(a[1], b[1])
    return inter / union if union > 0 else 0.0


def _x_overlaps(a, b) -> bool:
    return min(a[0] + a[2], b[0] + b[2]) > max(a[0], b[0])


def _norm(s: str) -> str:
    return " ".join(s.upper().split())


def is_prefix(a: str, b: str) -> bool:
    """True when the shorter text starts the longer one (typewriter reveal)."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    short, long_ = sorted((na, nb), key=len)
    return long_.startswith(short)


def _match(last: dict, d: dict) -> float:
    prefix = is_prefix(last["text"], d["text"])
    if not (prefix or fuzz.ratio(_norm(last["text"]), _norm(d["text"])) >= TEXT_MIN):
        return 0.0
    overlap = iou(last["box"], d["box"])
    if overlap >= IOU_MIN:
        return overlap
    if prefix and _y_iou(last["box"], d["box"]) >= 0.5 and _x_overlaps(last["box"], d["box"]):
        return max(overlap, 0.01)
    return 0.0


def _summarise(obs: list[dict]) -> dict:
    counts = Counter(o["text"] for o in obs)
    text = max(counts, key=lambda t: (counts[t], len(t)))
    boxes = np.array([o["box"] for o in obs if o["text"] == text])
    return {"text": text, "start_ms": obs[0]["t_ms"], "end_ms": obs[-1]["t_ms"],
            "box": [float(v) for v in np.median(boxes, axis=0)], "obs": obs}


def build_tracks(samples: list[dict], sample_ms: int) -> list[dict]:
    active: list[list[dict]] = []
    done: list[list[dict]] = []
    for si, s in enumerate(samples):
        pairs = sorted(((_match(tr[-1], d), ti, di) for ti, tr in enumerate(active)
                        for di, d in enumerate(s["dets"])), reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score <= 0 or ti in used_t or di in used_d:
                continue
            active[ti].append({**s["dets"][di], "t_ms": s["t_ms"], "frame": s["frame"], "si": si})
            used_t.add(ti)
            used_d.add(di)
        for di, d in enumerate(s["dets"]):
            if di not in used_d:
                active.append([{**d, "t_ms": s["t_ms"], "frame": s["frame"], "si": si}])
        still = []
        for tr in active:
            (still if si - tr[-1]["si"] <= MAX_GAP else done).append(tr)
        active = still
    done += active
    tracks = [_summarise(obs) for obs in done if obs[-1]["t_ms"] - obs[0]["t_ms"] + sample_ms >= MIN_TRACK_MS]
    tracks.sort(key=lambda t: (t["start_ms"], t["box"][1]))
    for i, t in enumerate(tracks):
        t["id"] = f"t{i + 1}"
    return tracks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_track.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/text/track.py tests/unit/test_text_track.py
git commit -m "feat: link OCR detections into text tracks"
```

---

### Task 6: Exact start/end frames (pure)

**Files:**
- Create: `vtpl/analyzers/text/refine.py`
- Test: `tests/unit/test_text_refine.py`

Background: OCR samples are ~133 ms apart. Inside a window around the first/last sample we compare each full-rate crop to an *empty* crop (window edge, assumed text-free) and the confirmed *text* crop. A frame "has text" once it is at least `FRAC` of the way from empty to text — so a fade-in is caught early, not half-way.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from vtpl.analyzers.text.refine import refine_end, refine_start, text_flags

EMPTY = np.zeros((10, 40, 3), np.uint8)
TEXT = np.full((10, 40, 3), 200, np.uint8)


def ramp(alphas):
    return [(EMPTY * (1 - a) + TEXT * a).astype(np.uint8) for a in alphas]


def test_text_flags_threshold():
    assert text_flags(ramp([0, 0.1, 0.2, 1]), TEXT, EMPTY) == [False, False, True, True]


def test_refine_start_walks_back_to_first_visible_frame():
    crops = ramp([0, 0, 0, 0.3, 0.6, 1, 1])          # frames 10..16, confirmed OCR sample at 15
    ids = list(range(10, 17))
    assert refine_start(ids, crops, 5, TEXT, EMPTY) == 13


def test_refine_end_walks_forward_to_last_visible_frame():
    crops = ramp([1, 1, 1, 0.5, 0.2, 0, 0])          # frames 40..46, confirmed at 41
    ids = list(range(40, 47))
    assert refine_end(ids, crops, 1, TEXT, EMPTY) == 44


def test_confirmed_frame_without_text_is_kept():
    assert refine_start([1, 2, 3], ramp([0, 0, 0]), 2, TEXT, EMPTY) == 3


def test_identical_refs_mean_text_throughout():
    assert refine_start([5, 6, 7], ramp([1, 1, 1]), 2, TEXT, TEXT) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_refine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.text.refine'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/text/refine.py`

```python
"""Find the exact first/last frame of a text track from full-rate crops around an OCR sample."""
import numpy as np

FRAC = 0.15      # a frame has text once it is 15 % of the way from the empty crop to the text crop


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def text_flags(crops: list[np.ndarray], text_ref: np.ndarray, empty_ref: np.ndarray, frac: float = FRAC) -> list[bool]:
    full = _dist(text_ref, empty_ref)
    if full < 1e-6:
        return [True] * len(crops)
    return [_dist(c, empty_ref) >= frac * full for c in crops]


def refine_start(ids: list[int], crops: list[np.ndarray], confirmed: int,
                 text_ref: np.ndarray, empty_ref: np.ndarray) -> int:
    """First frame of the run of text frames that ends at ids[confirmed]."""
    flags = text_flags(crops, text_ref, empty_ref)
    j = confirmed
    if not flags[j]:
        return ids[j]
    while j > 0 and flags[j - 1]:
        j -= 1
    return ids[j]


def refine_end(ids: list[int], crops: list[np.ndarray], confirmed: int,
               text_ref: np.ndarray, empty_ref: np.ndarray) -> int:
    """Last frame of the run of text frames that starts at ids[confirmed]."""
    flags = text_flags(crops, text_ref, empty_ref)
    j = confirmed
    if not flags[j]:
        return ids[j]
    while j < len(ids) - 1 and flags[j + 1]:
        j += 1
    return ids[j]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_refine.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/text/refine.py tests/unit/test_text_refine.py
git commit -m "feat: refine text start and end to the exact frame"
```

---

### Task 7: Text style from a crop (pure)

**Files:**
- Create: `vtpl/analyzers/text/style.py`
- Test: `tests/unit/test_text_style.py`

Background: k-means (k=4, Lab) splits the crop into colour clusters. Clusters common on the crop border are background. Of the rest, the **fill** is the cluster deepest inside the text (largest distance from background). A remaining cluster is a **stroke** if it rings the fill on all sides and is at least 2 px thick; it is a **shadow** if it sits only below-right of the text and is darker than the background. Anti-alias clusters (colours on the line between fill and background) are ignored.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from tests.synth.text_fixtures import text_crop
from vtpl.analyzers.text.style import analyze_style, combine_styles, delta_e, hex_from_bgr, ink_height, to_lab


def lab_of_hex(h):
    rgb = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return to_lab(np.array([rgb[::-1]], np.uint8))[0]


def close(hex_a, hex_b, tol=15):
    return delta_e(lab_of_hex(hex_a), lab_of_hex(hex_b)) <= tol


def test_white_fill_black_stroke():
    s = analyze_style(text_crop("WAIT FOR IT", "Montserrat-Bold", 96, (255, 255, 255), stroke=((0, 0, 0), 6)))
    assert close(hex_from_bgr(s.color_bgr), "#FFFFFF")
    assert s.stroke_bgr is not None and close(hex_from_bgr(s.stroke_bgr), "#000000") and 4 <= s.stroke_px <= 8
    assert not s.shadow


def test_yellow_with_shadow_no_stroke():
    s = analyze_style(text_crop("DAY 1", "Anton-Regular", 110, (255, 214, 0), shadow=True))
    assert close(hex_from_bgr(s.color_bgr), "#FFD600") and s.stroke_bgr is None and s.shadow


def test_red_fill_white_stroke():
    s = analyze_style(text_crop("SALE", "BebasNeue-Regular", 160, (230, 30, 60), stroke=((255, 255, 255), 4)))
    assert close(hex_from_bgr(s.color_bgr), "#E61E3C")
    assert s.stroke_bgr is not None and close(hex_from_bgr(s.stroke_bgr), "#FFFFFF")


def test_plain_text_has_no_stroke_from_antialiasing():
    s = analyze_style(text_crop("hello", "Poppins-Regular", 90, (255, 255, 255)))
    assert s.stroke_bgr is None and not s.shadow


def test_ink_height_and_combine():
    s = analyze_style(text_crop("SALE", "BebasNeue-Regular", 160, (230, 30, 60)))
    assert 100 <= ink_height(s.fill_mask) <= 130            # Bebas cap height is ~0.7 em
    combined = combine_styles([s, s, s])
    assert combined["stroke"] is None and combined["shadow"] is False and combined["color"].startswith("#")


def test_solid_crop_raises():
    with pytest.raises(ValueError):
        analyze_style(np.full((60, 200, 3), 90, np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_style.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.text.style'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/text/style.py`

```python
"""Fill colour, stroke, shadow and glyph mask of one text crop (BGR uint8)."""
from dataclasses import dataclass

import cv2
import numpy as np

K = 4
BORDER_SHARE = 0.15        # clusters with this share of border pixels are background
MIN_SHARE = 0.02           # clusters smaller than this share of the crop are ignored
SAME_COLOR = 12            # Lab distance at which clusters are merged into the fill
STROKE_DIFF = 20


@dataclass
class Style:
    color_bgr: np.ndarray
    stroke_bgr: np.ndarray | None
    stroke_px: int
    shadow: bool
    fill_mask: np.ndarray


def to_lab(pixels_bgr: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 BGR -> (N, 3) float32 CIE L*a*b*."""
    lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab = lab.astype(np.float32)
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    return lab


def lab_image(img_bgr: np.ndarray) -> np.ndarray:
    return to_lab(img_bgr).reshape(img_bgr.shape[0], img_bgr.shape[1], 3)


def delta_e(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def hex_from_bgr(bgr) -> str:
    b, g, r = (int(round(float(v))) for v in bgr)
    return f"#{r:02X}{g:02X}{b:02X}"


def ink_height(mask: np.ndarray) -> int:
    rows = np.nonzero(mask.any(axis=1))[0]
    return int(rows[-1] - rows[0] + 1) if len(rows) else 0


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        mask[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
    return out


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def _clean(mask: np.ndarray) -> np.ndarray:
    """Drop specks: components smaller than 1 % of the mask's pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 0.01 * mask.sum()
    return keep[labels]


def _side_cover(base: np.ndarray, m: np.ndarray, d: int = 2) -> tuple[float, float]:
    """Share of the ring just above-left / below-right of `base` that is covered by `m`."""
    covers = []
    for dy, dx in ((-d, -d), (d, d)):
        ring = _shift(base, dy, dx) & ~base
        covers.append(float((ring & m).sum() / ring.sum()) if ring.any() else 0.0)
    return covers[0], covers[1]


def _ring_px(fill: np.ndarray, m: np.ndarray, max_r: int = 12) -> int:
    px = 0
    for r in range(1, max_r + 1):
        ring = _dilate(fill, r) & ~_dilate(fill, r - 1)
        if not ring.any() or (ring & m).sum() / ring.sum() < 0.5:
            break
        px = r
    return px


def _on_segment(c, a, b, tol: float) -> bool:
    c, a, b = (np.asarray(v, np.float32) for v in (c, a, b))
    ab = b - a
    t = float(np.clip(np.dot(c - a, ab) / max(float(np.dot(ab, ab)), 1e-6), 0, 1))
    return float(np.linalg.norm(c - (a + t * ab))) < tol


def analyze_style(crop: np.ndarray) -> Style:
    h, w = crop.shape[:2]
    data = lab_image(crop).reshape(-1, 3)
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    bg_ids = [k for k in range(K) if (border == k).mean() >= BORDER_SHARE]
    bg = np.isin(labels, bg_ids)
    cand = [k for k in range(K) if k not in bg_ids and (labels == k).mean() >= MIN_SHARE]
    if not bg.any() or not cand:
        raise ValueError("cannot separate text from background")
    depth = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    fill_id = max(cand, key=lambda k: float(depth[labels == k].mean()))
    fill = labels == fill_id
    for k in cand:
        if k != fill_id and delta_e(centers[k], centers[fill_id]) < SAME_COLOR:
            fill |= labels == k
    fill = _clean(fill)
    if not fill.any():
        raise ValueError("no text pixels found")
    bg_lab = np.median(data[bg.reshape(-1)], axis=0)
    fill_lab = centers[fill_id]
    others = [k for k in cand if delta_e(centers[k], fill_lab) >= SAME_COLOR]
    others.sort(key=lambda k: -int((labels == k).sum()))

    stroke_bgr, stroke_px, shadow = None, 0, False
    for k in others:
        if delta_e(centers[k], fill_lab) < STROKE_DIFF or _on_segment(centers[k], fill_lab, bg_lab, 10):
            continue
        m = labels == k
        if min(_side_cover(fill, m)) >= 0.5:
            px = _ring_px(fill, m)
            if px >= 2:
                stroke_bgr, stroke_px = np.median(crop[m], axis=0), px
                break
    text = fill | (_dilate(fill, stroke_px) if stroke_bgr is not None else fill)
    for k in others:
        m = labels == k
        tl, br = _side_cover(text, m)
        if br >= 0.5 and tl < 0.3 and centers[k][0] < bg_lab[0] - 10:
            shadow = True
            break
    return Style(np.median(crop[fill], axis=0), stroke_bgr, stroke_px, shadow, fill)


def combine_styles(styles: list[Style]) -> dict:
    """Majority vote / median over several frames of the same text."""
    strokes = [s for s in styles if s.stroke_bgr is not None]
    stroke = None
    if len(strokes) * 2 > len(styles):
        stroke = {"color": hex_from_bgr(np.median([s.stroke_bgr for s in strokes], axis=0)),
                  "px": int(np.median([s.stroke_px for s in strokes]))}
    return {"color": hex_from_bgr(np.median([s.color_bgr for s in styles], axis=0)),
            "stroke": stroke,
            "shadow": sum(s.shadow for s in styles) * 2 > len(styles)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_style.py -v`
Expected: 6 passed. If a case fails, print the k-means centres, border shares and side covers for that crop and adjust only the named constants (`BORDER_SHARE`, `MIN_SHARE`, `SAME_COLOR`, `STROKE_DIFF`) or the `_side_cover` distance `d`; do not change the tests' ground truth.

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/text/style.py tests/unit/test_text_style.py
git commit -m "feat: text fill, stroke and shadow from a crop"
```

---

### Task 8: Font ranking and em size (pure)

**Files:**
- Create: `vtpl/analyzers/text/font.py`
- Test: `tests/unit/test_text_font.py`

Background: render the OCR'd text in every catalog font at the measured ink height and compare glyph masks: score = 0.5·IoU + 0.5·(1 − chamfer distance / (0.1 × height)).

- [ ] **Step 1: Write the failing test**

```python
import cv2
import numpy as np
import pytest

from tests.synth.text_fixtures import text_crop
from vtpl.analyzers.text.font import crop_to_ink, em_size_for, ink_mask, rank_fonts, similarity
from vtpl.analyzers.text.style import analyze_style, ink_height


def video_like(crop):
    """Blur + JPEG round-trip, like a frame decoded from a compressed video."""
    blurred = cv2.GaussianBlur(crop, (3, 3), 0.8)
    ok, buf = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_crop_to_ink_and_ink_mask_height():
    m = np.zeros((20, 30), bool)
    m[5:9, 3:7] = True
    assert crop_to_ink(m).shape == (4, 4)
    assert ink_mask("SALE", "BebasNeue-Regular", 57).shape[0] == 57


def test_similarity_identity_beats_other_font():
    a = ink_mask("WAIT FOR IT", "Montserrat-Bold", 60)
    assert similarity(a, a) == pytest.approx(1.0)
    assert similarity(a, ink_mask("WAIT FOR IT", "PlayfairDisplay-Bold", 60)) < 0.8


@pytest.mark.parametrize("text,font_id,size", [
    ("WAIT FOR IT", "Montserrat-Bold", 96),
    ("SALE TODAY", "BebasNeue-Regular", 140),
    ("Golden Hour", "PlayfairDisplay-Bold", 100),
    ("day one", "Pacifico-Regular", 100),
])
def test_true_font_in_top3(text, font_id, size):
    s = analyze_style(video_like(text_crop(text, font_id, size, (255, 255, 255), stroke=((0, 0, 0), 4))))
    ranked = rank_fonts(text, s.fill_mask)
    assert len(ranked) == 3 and font_id in [r[0] for r in ranked], ranked


def test_em_size_recovered():
    s = analyze_style(text_crop("WAIT FOR IT", "Montserrat-Bold", 96, (255, 255, 255)))
    assert abs(em_size_for("WAIT FOR IT", "Montserrat-Bold", ink_height(s.fill_mask)) - 96) <= 4


def test_empty_mask_raises():
    with pytest.raises(ValueError, match="empty text mask"):
        rank_fonts("X", np.zeros((10, 10), bool))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_font.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.text.font'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/text/font.py`

```python
"""Guess a text's font by rendering it in every catalog font and comparing glyph masks."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vtpl.fontlib import catalog, pil_font

RENDER_SIZE = 200


def crop_to_ink(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    if not len(rows):
        return mask[:0, :0]
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


@lru_cache(maxsize=4096)
def _glyphs(text: str, font_id: str) -> np.ndarray:
    font = pil_font(font_id, RENDER_SIZE)
    x0, y0, x1, y1 = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
    img = Image.new("L", (x1 - x0 + 8, y1 - y0 + 8), 0)
    ImageDraw.Draw(img).text((4 - x0, 4 - y0), text, font=font, fill=255)
    return crop_to_ink(np.array(img) > 127)


def ink_mask(text: str, font_id: str, ink_h: int) -> np.ndarray:
    g = _glyphs(text, font_id)
    w = max(1, round(g.shape[1] * ink_h / g.shape[0]))
    return cv2.resize(g.astype(np.float32), (w, ink_h), interpolation=cv2.INTER_AREA) > 0.5


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    A, B = np.zeros((h, w), bool), np.zeros((h, w), bool)
    A[:a.shape[0], :a.shape[1]] = a
    B[:b.shape[0], :b.shape[1]] = b
    union = (A | B).sum()
    overlap = (A & B).sum() / union if union else 0.0
    if not (A.any() and B.any()):
        return 0.5 * overlap
    da = cv2.distanceTransform((~A).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~B).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = (da[B].mean() + db[A].mean()) / 2
    return float(0.5 * overlap + 0.5 * max(0.0, 1 - chamfer / (0.1 * h)))


def rank_fonts(text: str, fill_mask: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
    target = crop_to_ink(fill_mask)
    if target.size == 0:
        raise ValueError("empty text mask")
    scored = [(e.id, similarity(target, ink_mask(text, e.id, target.shape[0]))) for e in catalog()]
    return sorted(scored, key=lambda s: s[1], reverse=True)[:top]


def em_size_for(text: str, font_id: str, ink_h: int) -> int:
    """Pillow em size (px) at which `text` in this font has the given ink height."""
    return max(1, round(RENDER_SIZE * ink_h / _glyphs(text, font_id).shape[0]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_font.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/text/font.py tests/unit/test_text_font.py
git commit -m "feat: rank catalog fonts by rendered glyph similarity"
```

---

### Task 9: Animation measurement and classification (pure)

**Files:**
- Create: `vtpl/analyzers/text/anim.py`
- Test: `tests/unit/test_text_anim.py`

Background: crops are a full-width *band* around the text (so slides stay in view). Per frame we find pixels matching the fill colour and measure their bounding box (w, h, centre) plus overall contrast against the empty band. The sequence starts at the refined start (or, reversed, at the refined end) and its last element is the settled text.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tests.synth.text_fixtures import TextItem, frame_at
from vtpl.analyzers.text.anim import classify, measure

STEP = 1000 / 30
BAND = slice(860, 1060)


def band(rgb):
    return rgb[BAND, :, ::-1].copy()


def seq_in(item, n=10):
    crops = [band(frame_at([item], round(item.start_ms + k * STEP))) for k in range(n)]
    return crops, band(frame_at([], 0))


def seq_out(item, n=10):
    crops = [band(frame_at([item], round(item.end_ms - 1 - k * STEP))) for k in range(n)]
    return crops, band(frame_at([], 0))


def item(**kw):
    base = dict(text="HELLO", font_id="Montserrat-Bold", size=110, color=(255, 255, 255), center=(540, 960),
                start_ms=0, end_ms=3000, anim_ms=200)
    return TextItem(**{**base, **kw})


@pytest.mark.parametrize("anim", ["pop", "fade", "slide", "typewriter", "none"])
def test_in_animations(anim):
    it = item(anim_in=anim, anim_ms=300 if anim == "typewriter" else 200)
    crops, empty = seq_in(it)
    res = classify(measure(crops, (255, 255, 255), empty), 1080, 1920, STEP)
    assert res["kind"] == anim
    if anim == "slide":
        assert res["from"]["dx"] == pytest.approx(-300 / 1080, abs=0.05) and abs(res["from"]["dy"]) < 0.01
    if anim != "none":
        assert 100 <= res["ms"] <= 400


@pytest.mark.parametrize("anim", ["pop", "fade", "none"])
def test_out_animations(anim):
    crops, empty = seq_out(item(anim_out=anim))
    assert classify(measure(crops, (255, 255, 255), empty), 1080, 1920, STEP)["kind"] == anim


def test_no_text_is_none():
    crops = [band(frame_at([], 0))] * 5
    assert classify(measure(crops, (255, 255, 255), crops[0]), 1080, 1920, STEP) == {"kind": "none", "ms": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_anim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.text.anim'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/text/anim.py`

```python
"""Classify a text's in/out animation from a short sequence of band crops."""
import cv2
import numpy as np

from vtpl.analyzers.text.style import lab_image, to_lab

FILL_TOL = 18.0          # Lab distance for a pixel to count as fill colour
MIN_PIXELS = 20


def measure(crops: list[np.ndarray], fill_bgr, empty: np.ndarray) -> list[dict]:
    fill_lab = to_lab(np.asarray(fill_bgr, np.float32).round().astype(np.uint8).reshape(1, 3))[0]
    out = []
    for c in crops:
        m = np.linalg.norm(lab_image(c) - fill_lab, axis=2) < FILL_TOL
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        contrast = float(np.abs(c.astype(np.float32) - empty.astype(np.float32)).mean())
        if m.sum() < MIN_PIXELS:
            out.append({"n": 0, "w": 0.0, "h": 0.0, "cx": 0.0, "cy": 0.0, "contrast": contrast})
            continue
        ys, xs = np.nonzero(m)
        x0, x1 = np.percentile(xs, [1, 99])
        y0, y1 = np.percentile(ys, [1, 99])
        out.append({"n": int(m.sum()), "w": float(x1 - x0 + 1), "h": float(y1 - y0 + 1),
                    "cx": float((x0 + x1) / 2), "cy": float((y0 + y1) / 2), "contrast": contrast})
    return out


def _nondecreasing(xs: list[float], tol: float) -> bool:
    return all(b >= a - tol for a, b in zip(xs, xs[1:]))


def _settled(m: dict, ref: dict, frame_w: int) -> bool:
    return (m["n"] > 0 and abs(m["h"] / ref["h"] - 1) <= 0.03 and abs(m["w"] / ref["w"] - 1) <= 0.03
            and abs(m["cx"] - ref["cx"]) / frame_w <= 0.01 and m["contrast"] >= 0.95 * ref["contrast"])


def classify(ms: list[dict], frame_w: int, frame_h: int, step_ms: float) -> dict:
    """{"kind": none|pop|fade|slide|typewriter, "ms": duration, ["from": {"dx", "dy"}]}; ms[-1] is settled."""
    ref = ms[-1]
    valid = [m for m in ms if m["n"] > 0]
    if ref["n"] == 0 or not valid:
        return {"kind": "none", "ms": 0}
    hr = [m["h"] / ref["h"] for m in valid]
    wr = [m["w"] / ref["w"] for m in valid]
    dx = (valid[0]["cx"] - ref["cx"]) / frame_w
    dy = (valid[0]["cy"] - ref["cy"]) / frame_h
    cr = [m["contrast"] / ref["contrast"] if ref["contrast"] > 0 else 1.0 for m in ms]
    extra = {}
    if len(valid) >= 3 and wr[0] <= 0.6 and all(abs(h - 1) <= 0.1 for h in hr) and _nondecreasing(wr, 0.05):
        kind = "typewriter"
    elif (abs(dx) >= 0.05 or abs(dy) >= 0.05) and all(abs(h - 1) <= 0.15 for h in hr):
        kind, extra = "slide", {"from": {"dx": round(dx, 3), "dy": round(dy, 3)}}
    elif max(hr) >= 1.05 or hr[0] <= 0.9:
        kind = "pop"
    elif cr[0] <= 0.6 and _nondecreasing(cr, 0.05):
        kind = "fade"
    else:
        return {"kind": "none", "ms": 0}
    settle = next((i for i, m in enumerate(ms) if _settled(m, ref, frame_w)), len(ms) - 1)
    return {"kind": kind, "ms": int(min(400, max(100, round(settle * step_ms)))), **extra}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_text_anim.py -v`
Expected: 9 passed. If one kind is misclassified, print `measure()` output for that sequence and adjust only the thresholds in `classify` (not the fixture or expected labels).

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/text/anim.py tests/unit/test_text_anim.py
git commit -m "feat: classify text in and out animations"
```

---

### Task 10: Text stage orchestration and pipeline registration

**Files:**
- Create: `vtpl/analyzers/text/stage.py`
- Modify: `vtpl/pipeline.py`, `tests/unit/test_pipeline.py:19-20`
- Test: `tests/integration/test_text_stage.py`

- [ ] **Step 1: Write the failing tests**

`tests/integration/test_text_stage.py`:
```python
import numpy as np
import pytest
from rapidfuzz import fuzz

from vtpl import frames, ingest
from vtpl.analyzers.text import stage
from vtpl.analyzers.text.style import delta_e, to_lab
from vtpl.util.workdir import WorkDir

pytestmark = pytest.mark.slow


def lab(h):
    return to_lab(np.array([[int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)]], np.uint8))[0]


@pytest.fixture(scope="module")
def tracks(tmp_path_factory, text_video_basic):
    wd = WorkDir(tmp_path_factory.mktemp("text") / "w")
    ingest.run(wd, text_video_basic["path"])
    frames.run(wd, text_video_basic["path"])
    assert stage.run(wd, text_video_basic["path"]) == []
    return wd.read_json("text.json")["tracks"]


def test_one_track_per_text(tracks, text_video_basic):
    truth = text_video_basic["truth"]
    assert len(tracks) == len(truth)
    for t in truth:
        assert any(fuzz.ratio(tr["text"].upper(), t["text"]) >= 80 for tr in tracks), t["text"]


def _match(tracks, t):
    return max(tracks, key=lambda tr: fuzz.ratio(tr["text"].upper(), t["text"]))


def test_timing_position_colour_stroke_shadow(tracks, text_video_basic):
    for t in text_video_basic["truth"]:
        tr = _match(tracks, t)
        assert abs(tr["start_ms"] - t["start_ms"]) <= 100, (t["text"], tr["start_ms"])
        assert abs(tr["end_ms"] - t["end_ms"]) <= 100, (t["text"], tr["end_ms"])
        b = tr["box"]
        assert abs(b["x"] + b["w"] / 2 - t["center"][0]) <= 0.02
        assert abs(b["y"] + b["h"] / 2 - t["center"][1]) <= 0.02
        assert delta_e(lab(tr["style"]["color"]), lab(t["color"])) <= 15
        assert (tr["style"]["stroke"] is not None) == t["stroke"]
        assert tr["style"]["shadow"] == t["shadow"]


def test_font_size_and_animation(tracks, text_video_basic):
    for t in text_video_basic["truth"]:
        tr = _match(tracks, t)
        assert t["font"] in [c[0] for c in tr["font"]["candidates"]], (t["text"], tr["font"])
        if tr["font"]["id"] == t["font"]:
            assert abs(tr["style"]["size_px"] - t["size"]) <= 0.08 * t["size"]
        assert tr["anim"]["in"] == t["anim_in"], (t["text"], tr["anim"])
        assert tr["anim"]["out"] == t["anim_out"], (t["text"], tr["anim"])
```

In `tests/unit/test_pipeline.py` change `test_default_stage_order` to:
```python
def test_default_stage_order():
    assert STAGE_NAMES == ["ingest", "frames", "shots", "audio", "speech", "text", "merge"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/user/.venv/bin/pytest tests/integration/test_text_stage.py tests/unit/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'stage'` and the stage-order assertion.

- [ ] **Step 3: Implement** — `vtpl/analyzers/text/stage.py`

```python
"""Text stage: OCR samples -> tracks -> exact timing, style, font, size and animation per track."""
from collections import defaultdict
from pathlib import Path

import cv2

from vtpl import frames as fr
from vtpl.analyzers.text import anim, detect, font, refine, style, track
from vtpl.util.timebase import frame_to_ms, ms_to_frame
from vtpl.util.workdir import WorkDir

OUTPUT = "text.json"
WINDOW_MS = 200       # searched before the first / after the last OCR sample
ANIM_MS = 300         # animations are measured over this long
STYLE_FRAMES = 3
STYLE_PAD = 0.25      # style crops extend the box by this share of its height


def _px(box, w, h):
    x, y, bw, bh = box
    return round(x * w), round(y * h), round(bw * w), round(bh * h)


def _expand(r, frac):
    x, y, w, h = r
    p = round(h * frac)
    return x - p, y - p, w + 2 * p, h + 2 * p


def _band(r, frame_w):
    x, y, w, h = r
    return 0, y - h, frame_w, 3 * h


def _style_frames(wd: WorkDir, t: dict, cache: dict, ratio: int) -> list[int]:
    """The sharpest STYLE_FRAMES sample frames, avoiding the animated first/last ANIM_MS."""
    inner = [o for o in t["obs"] if t["start_ms"] + ANIM_MS <= o["t_ms"] <= t["end_ms"] - ANIM_MS] or t["obs"]
    x, y, w, h = fr.clamp_rect(_px(t["box"], cache["w"], cache["h"]), cache["w"], cache["h"])

    def sharpness(o):
        g = cv2.cvtColor(fr.cached_frame(wd, o["frame"] // ratio)[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(g, cv2.CV_64F).var() if g.size else 0.0

    return [o["frame"] for o in sorted(inner, key=sharpness, reverse=True)[:STYLE_FRAMES]]


def run(wd: WorkDir, source: Path) -> list[str]:
    meta = wd.read_json("meta.json")
    if not wd.has(fr.OUTPUT):
        raise FileNotFoundError(f"{fr.OUTPUT} missing: run the frames stage first")
    cache = wd.read_json(fr.OUTPUT)
    W, H, fps = meta["w"], meta["h"], meta["fps"]
    ratio = round(fps / cache["fps"])
    tracks = track.build_tracks(detect.detect_samples(wd), round(detect.SAMPLE_EVERY * 1000 / cache["fps"]))
    if not tracks:
        wd.write_json(OUTPUT, {"tracks": []})
        return ["no on-screen text found"]

    last = ms_to_frame(meta["duration_ms"], fps) - 1
    win, an = ms_to_frame(WINDOW_MS, fps), ms_to_frame(ANIM_MS, fps)
    requests: dict[int, list] = defaultdict(list)
    plans = {}
    for t in tracks:
        r = _px(t["box"], W, H)
        s0, s1 = t["obs"][0]["frame"], t["obs"][-1]["frame"]
        start_ids = list(range(max(0, s0 - win), min(last, s0 + an) + 1))
        end_ids = list(range(max(0, s1 - an), min(last, s1 + win) + 1))
        style_ids = _style_frames(wd, t, cache, ratio)
        for f in sorted(set(start_ids) | set(end_ids)):
            requests[f].append((t["id"] + ":band", _band(r, W)))
        for f in style_ids:
            requests[f].append((t["id"] + ":style", _expand(r, STYLE_PAD)))
        plans[t["id"]] = (start_ids, end_ids, style_ids, s0, s1)
    crops = fr.read_crops(wd.path("norm.mp4"), dict(requests))

    out, warnings = [], []
    for t in tracks:
        try:
            out.append(_analyse(t, plans[t["id"]], crops, meta))
        except Exception as e:
            warnings.append(f"{t['id']} ({t['text']!r}) skipped: {e}")
    wd.write_json(OUTPUT, {"tracks": out})
    return warnings


def _analyse(t: dict, plan, crops: dict, meta: dict) -> dict:
    start_ids, end_ids, style_ids, s0, s1 = plan
    W, H, fps = meta["w"], meta["h"], meta["fps"]
    band = crops[t["id"] + ":band"]
    start_crops, end_crops = [band[f] for f in start_ids], [band[f] for f in end_ids]
    f_start = refine.refine_start(start_ids, start_crops, start_ids.index(s0), band[s0], start_crops[0])
    f_end = refine.refine_end(end_ids, end_crops, end_ids.index(s1), band[s1], end_crops[-1])

    styles = [style.analyze_style(crops[t["id"] + ":style"][f]) for f in style_ids]
    best_style = styles[0]                                   # style_ids are sharpest first
    ranked = font.rank_fonts(t["text"], best_style.fill_mask)
    font_id, confidence = ranked[0]
    size_px = font.em_size_for(t["text"], font_id, style.ink_height(best_style.fill_mask))

    an, step = ms_to_frame(ANIM_MS, fps), 1000 / fps
    in_ids = [f for f in start_ids if f_start <= f <= f_start + an]
    out_ids = [f for f in reversed(end_ids) if f_end - an <= f <= f_end]
    a_in = anim.classify(anim.measure([band[f] for f in in_ids], best_style.color_bgr, start_crops[0]), W, H, step)
    a_out = anim.classify(anim.measure([band[f] for f in out_ids], best_style.color_bgr, end_crops[-1]), W, H, step)
    anim_doc = {"in": a_in["kind"], "in_ms": a_in["ms"], "out": a_out["kind"], "out_ms": a_out["ms"]}
    if "from" in a_in:
        anim_doc["in_from"] = a_in["from"]
    if "from" in a_out:
        anim_doc["out_to"] = a_out["from"]

    x, y, w, h = t["box"]
    return {
        "id": t["id"], "text": t["text"],
        "start_ms": frame_to_ms(f_start, fps),
        "end_ms": min(meta["duration_ms"], frame_to_ms(f_end + 1, fps)),
        "box": {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)},
        "style": {**style.combine_styles(styles), "size_px": size_px},
        "font": {"id": font_id, "confidence": round(confidence, 3),
                 "candidates": [[fid, round(score, 3)] for fid, score in ranked]},
        "anim": anim_doc,
    }
```

In `vtpl/pipeline.py`, replace the imports and `STAGES`:
```python
from vtpl import frames, ingest, merge
from vtpl.analyzers import audio, shots, speech
from vtpl.analyzers.text import stage as text_stage
```
```python
STAGES = [
    Stage("ingest", ingest.OUTPUT, ingest.run),
    Stage("frames", frames.OUTPUT, frames.run),
    Stage("shots", shots.OUTPUT, shots.run),
    Stage("audio", audio.OUTPUT, audio.run),
    Stage("speech", speech.OUTPUT, speech.run),
    Stage("text", text_stage.OUTPUT, text_stage.run),
    Stage("merge", merge.OUTPUT, merge.run),
]
```
(`CRITICAL` stays `{"ingest", "shots", "merge"}` — frames and text are optional.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/user/.venv/bin/pytest tests/integration/test_text_stage.py tests/unit/test_pipeline.py -v`
Expected: all pass. If a tolerance fails, print the track and the truth, find which step is off (detect / track / refine / style / font / anim) and fix that module within its design; add a unit test for the case to that module's test file. Do not loosen these tolerances.

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/text/stage.py vtpl/pipeline.py tests/integration/test_text_stage.py tests/unit/test_pipeline.py
git commit -m "feat: text stage extracts overlays with timing, style, font and animation"
```

---

### Task 11: Template text items, caption dedupe, schema

**Files:**
- Modify: `vtpl/schema.py`, `vtpl/merge.py`
- Test: `tests/unit/test_merge_texts.py`, `tests/unit/test_schema.py` (append)

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_merge_texts.py`:
```python
import copy

import pytest

from vtpl import merge
from vtpl.schema import TemplateError, validate

META = {"w": 1080, "h": 1920, "fps": 30, "duration_ms": 4000, "has_audio": True}
SHOTS = {"shots": [{"start_ms": 0, "end_ms": 4000, "transition_in": {"type": "cut", "duration_ms": 0}}]}


def track(tid, text, start, end, conf=0.8):
    return {"id": tid, "text": text, "start_ms": start, "end_ms": end,
            "box": {"x": 0.1, "y": 0.6, "w": 0.8, "h": 0.05},
            "style": {"color": "#FFFFFF", "stroke": {"color": "#000000", "px": 5}, "shadow": False, "size_px": 90},
            "font": {"id": "BebasNeue-Regular", "confidence": conf, "candidates": [["BebasNeue-Regular", conf]]},
            "anim": {"in": "pop", "in_ms": 150, "out": "fade", "out_ms": 200}}


WORDS = [{"w": "so", "start_ms": 1000, "end_ms": 1200, "p": 0.9}, {"w": "today", "start_ms": 1200, "end_ms": 1600, "p": 0.9},
         {"w": "we", "start_ms": 1700, "end_ms": 1800, "p": 0.9}]


def test_is_caption():
    assert merge.is_caption(track("t1", "SO TODAY", 1000, 1600), WORDS)
    assert not merge.is_caption(track("t1", "WAIT FOR IT", 1000, 1600), WORDS)
    assert not merge.is_caption(track("t1", "SO TODAY", 3000, 3500), WORDS)     # wrong time


def test_texts_overlays_slots_and_caption_ref():
    tracks = [track("t1", "SO TODAY", 1000, 1600), track("t2", "SALE", 500, 2500), track("t3", "DAY 1", 2000, 3000)]
    t = merge.build_template(META, SHOTS, None, {"words": WORDS}, "abc", {"tracks": tracks})
    validate(t)
    overlays = [x for x in t["texts"] if x["kind"] == "overlay"]
    assert [(x["id"], x["slot"], x["sample"]) for x in overlays] == [("t1", "text_1", "SALE"), ("t2", "text_2", "DAY 1")]
    cap = next(x for x in t["texts"] if x["kind"] == "caption")
    assert cap["id"] == "c1" and t["captions"]["style_ref"] == "c1" and cap["sample"] == "SO TODAY"
    assert overlays[0]["style"] == {"font": "BebasNeue-Regular", "font_confidence": 0.8, "size_px": 90,
                                    "color": "#FFFFFF", "stroke": {"color": "#000000", "px": 5}, "shadow": False}


def test_low_confidence_font_falls_back():
    t = merge.build_template(META, SHOTS, None, None, "abc", {"tracks": [track("t1", "SALE", 0, 1000, conf=0.3)]})
    st = t["texts"][0]["style"]
    assert st["font"] == "Montserrat-Bold" and st["font_guess"] == "BebasNeue-Regular"


def test_texts_clipped_to_duration_and_dropped_if_empty():
    tracks = [track("t1", "A LONG ONE", 3000, 4500), track("t2", "LATE", 4100, 4600)]
    t = merge.build_template(META, SHOTS, None, None, "abc", {"tracks": tracks})
    assert [(x["sample"], x["end_ms"]) for x in t["texts"]] == [("A LONG ONE", 4000)]


def test_run_reads_text_json(tmp_path):
    from vtpl.util.workdir import WorkDir
    wd = WorkDir(tmp_path / "abc")
    wd.write_json("meta.json", META)
    wd.write_json("shots.json", SHOTS)
    wd.write_json("text.json", {"tracks": [track("t1", "SALE", 0, 1000)]})
    merge.run(wd, tmp_path / "v.mp4")
    assert wd.read_json("template.json")["texts"][0]["sample"] == "SALE"
```

Append to `tests/unit/test_schema.py`:
```python
TEXT = {"id": "t1", "slot": "text_1", "sample": "HI", "kind": "overlay", "start_ms": 100, "end_ms": 900,
        "box": {"x": 0.1, "y": 0.5, "w": 0.3, "h": 0.05},
        "style": {"font": "Montserrat-Bold", "font_confidence": 0.9, "size_px": 80, "color": "#FFFFFF",
                  "stroke": None, "shadow": False},
        "anim": {"in": "fade", "in_ms": 200, "out": "none", "out_ms": 0}}


def test_text_item_valid_and_errors():
    t = copy.deepcopy(GOOD)
    t["texts"] = [copy.deepcopy(TEXT)]
    validate(t)
    t["texts"][0]["anim"]["in"] = "spin"
    with pytest.raises(TemplateError, match="texts/0/anim/in"):
        validate(t)
    t["texts"][0]["anim"]["in"] = "fade"
    t["texts"][0]["end_ms"] = 5000
    with pytest.raises(TemplateError, match=r"texts\[0\] must lie within"):
        validate(t)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/user/.venv/bin/pytest tests/unit/test_merge_texts.py tests/unit/test_schema.py -v`
Expected: FAIL — `build_template() takes 5 positional arguments but 6 were given`, `is_caption` missing, schema tests fail.

- [ ] **Step 3: Implement**

In `vtpl/schema.py`, add before `TEMPLATE_SCHEMA`:
```python
ANIMS = ["none", "pop", "fade", "slide", "typewriter"]
UNIT = {"type": "number", "minimum": 0, "maximum": 1}
OFFSET = {"type": "object", "required": ["dx", "dy"], "properties": {"dx": {"type": "number"}, "dy": {"type": "number"}}}
TEXT_ITEM = {
    "type": "object",
    "required": ["id", "slot", "sample", "kind", "start_ms", "end_ms", "box", "style", "anim"],
    "properties": {
        "id": {"type": "string"}, "slot": {"type": "string"}, "sample": {"type": "string"},
        "kind": {"enum": ["overlay", "caption"]}, "start_ms": INT, "end_ms": INT,
        "box": {"type": "object", "required": ["x", "y", "w", "h"],
                "properties": {"x": UNIT, "y": UNIT, "w": UNIT, "h": UNIT}},
        "style": {
            "type": "object",
            "required": ["font", "font_confidence", "size_px", "color", "stroke", "shadow"],
            "properties": {
                "font": {"type": "string"}, "font_guess": {"type": "string"},
                "font_confidence": {"type": "number"}, "size_px": {"type": "integer", "minimum": 1},
                "color": {"type": "string", "pattern": "^#[0-9A-F]{6}$"},
                "stroke": {"oneOf": [{"type": "null"}, {
                    "type": "object", "required": ["color", "px"],
                    "properties": {"color": {"type": "string", "pattern": "^#[0-9A-F]{6}$"},
                                   "px": {"type": "integer", "minimum": 0}}}]},
                "shadow": {"type": "boolean"},
            },
        },
        "anim": {"type": "object", "required": ["in", "in_ms", "out", "out_ms"],
                 "properties": {"in": {"enum": ANIMS}, "out": {"enum": ANIMS}, "in_ms": INT, "out_ms": INT,
                                "in_from": OFFSET, "out_to": OFFSET}},
    },
}
```
Change the `"texts"` entry in `TEMPLATE_SCHEMA["properties"]` to:
```python
        "texts": {"type": "array", "items": TEXT_ITEM},
```
At the end of `check_timeline`, add:
```python
    for i, x in enumerate(t["texts"]):
        if not 0 <= x["start_ms"] < x["end_ms"] <= dur:
            raise TemplateError(f"texts[{i}] must lie within 0..{dur} ms with start before end")
```

In `vtpl/merge.py`, add imports and helpers:
```python
from rapidfuzz import fuzz

from vtpl.fontlib import FALLBACK_ID

FONT_MIN_CONFIDENCE = 0.5
CAPTION_SHARE = 0.6        # share of a text's words that must match nearby speech to count as a caption
CAPTION_SLACK_MS = 200


def _tokens(s: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in s.upper()).split()


def is_caption(track: dict, words: list[dict]) -> bool:
    toks = _tokens(track["text"])
    if not toks:
        return False
    near = [tok for w in words
            if w["end_ms"] >= track["start_ms"] - CAPTION_SLACK_MS and w["start_ms"] <= track["end_ms"] + CAPTION_SLACK_MS
            for tok in _tokens(w["w"])]
    hits = sum(1 for tok in toks if any(fuzz.ratio(tok, n) >= 80 for n in near))
    return hits / len(toks) >= CAPTION_SHARE


def _text_item(track: dict, item_id: str, slot: str, kind: str, dur: int) -> dict:
    font, st = track["font"], track["style"]
    style_doc = {"font": font["id"], "font_confidence": font["confidence"], "size_px": st["size_px"],
                 "color": st["color"], "stroke": st["stroke"], "shadow": st["shadow"]}
    if font["confidence"] < FONT_MIN_CONFIDENCE:
        style_doc["font"], style_doc["font_guess"] = FALLBACK_ID, font["id"]
    return {"id": item_id, "slot": slot, "sample": track["text"], "kind": kind,
            "start_ms": track["start_ms"], "end_ms": min(track["end_ms"], dur),
            "box": track["box"], "style": style_doc, "anim": track["anim"]}


def build_texts(tracks: list[dict], words: list[dict], dur: int) -> tuple[list[dict], str | None]:
    overlays, caption = [], None
    for tr in sorted(tracks, key=lambda t: t["start_ms"]):
        if tr["start_ms"] >= dur:
            continue
        if words and is_caption(tr, words):
            if caption is None:
                caption = _text_item(tr, "c1", "captions", "caption", dur)
            continue
        n = len(overlays) + 1
        overlays.append(_text_item(tr, f"t{n}", f"text_{n}", "overlay", dur))
    return overlays + ([caption] if caption else []), ("c1" if caption else None)
```
Change `build_template`'s signature and its texts/captions lines:
```python
def build_template(meta: dict, shots_doc: dict, audio: dict | None, speech: dict | None, source_id: str,
                   text_doc: dict | None = None) -> dict:
```
```python
    texts, style_ref = build_texts((text_doc or {}).get("tracks", []), words, dur)
```
(place after `words` is computed), and in the template dict use:
```python
        "texts": texts,
        "captions": {"style_ref": style_ref, "words": words},
```
In `run`, read `text.json` like the other optional inputs:
```python
    for name in ("audio.json", "speech.json", "text.json"):
        if wd.has(name):
            optional[name] = wd.read_json(name)
        else:
            warnings.append(f"{name} missing: " + {"audio.json": "no beats", "speech.json": "no captions",
                                                   "text.json": "no text overlays"}[name])
    template = build_template(wd.read_json("meta.json"), wd.read_json("shots.json"),
                              optional.get("audio.json"), optional.get("speech.json"), source_id=wd.root.name,
                              text_doc=optional.get("text.json"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/user/.venv/bin/pytest tests/unit/test_merge_texts.py tests/unit/test_schema.py tests/unit/test_merge.py -v`
Expected: all pass. `tests/unit/test_merge.py::test_run_reads_workdir` asserts the warnings list — update its expected list to `["speech.json missing: no captions", "text.json missing: no text overlays"]` (the new warning is intended).

- [ ] **Step 5: Commit**

```bash
git add vtpl/schema.py vtpl/merge.py tests/unit/test_merge_texts.py tests/unit/test_schema.py tests/unit/test_merge.py
git commit -m "feat: text items in template with caption dedupe and font fallback"
```

---

### Task 12: ASS document builder (pure)

**Files:**
- Create: `vtpl/render/ass.py`
- Modify: `vtpl/render/slots.py` (accept a `captions:` key)
- Test: `tests/unit/test_ass.py`, `tests/unit/test_slots.py` (append)

Background: ASS (libass) draws text at script resolution 1080×1920. Each overlay gets its own style; animations become override tags: pop `\fscx80\fscy80\t(...)`, fade `\fad`, slide `\move`, typewriter = one event per revealed prefix with the rest hidden by `\alpha&HFF&` (keeps the layout fixed). Captions are grouped into lines of up to 3 words and use karaoke `\k` timing.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_ass.py`:
```python
import copy

from tests.unit.test_schema import GOOD, TEXT
from vtpl.fontlib import ass_font_size
from vtpl.render.ass import (ass_color, ass_time, build_ass, caption_lines, escape, overlay_events,
                             replace_words)


def tpl(texts=None, words=None):
    t = copy.deepcopy(GOOD)
    t["texts"] = texts or []
    t["captions"]["words"] = words if words is not None else []
    return t


def item(**anim):
    x = copy.deepcopy(TEXT)
    x["anim"].update(anim)
    return x


def test_time_color_escape():
    assert ass_time(0) == "0:00:00.00" and ass_time(61234) == "0:01:01.23"
    assert ass_color("#FF8000") == "&H000080FF" and ass_color("#FFFFFF", 0x80) == "&H80FFFFFF"
    assert escape("a{b}\\c\nd") == "a(b)/c\\Nd"


def test_style_lines_use_family_and_line_height():
    doc = build_ass(tpl([item()]), {})
    style = next(l for l in doc.splitlines() if l.startswith("Style: t1,"))
    fields = style[len("Style: "):].split(",")
    assert len(fields) == 23 and fields[1] == "Montserrat" and fields[2] == str(ass_font_size("Montserrat-Bold", 80))
    assert fields[7] == "-1" and fields[18] == "5"            # bold, centre alignment


def test_fade_pop_slide_tags():
    ev = overlay_events(item(**{"in": "fade", "in_ms": 200, "out": "fade", "out_ms": 100}), "HI")
    assert len(ev) == 1 and "\\fad(200,100)" in ev[0] and "\\pos(270,1008)" in ev[0]
    ev = overlay_events(item(**{"in": "pop", "in_ms": 150}), "HI")
    assert "\\fscx80\\fscy80\\t(0,100,\\fscx115\\fscy115)\\t(100,150,\\fscx100\\fscy100)" in ev[0]
    ev = overlay_events(item(**{"in": "slide", "in_ms": 200, "in_from": {"dx": -0.25, "dy": 0}}), "HI")
    assert "\\move(0,1008,270,1008,0,200)" in ev[0]


def test_typewriter_one_event_per_char_layout_fixed():
    ev = overlay_events(item(**{"in": "typewriter", "in_ms": 300}), "ABC")
    assert len(ev) == 3
    assert ev[0].split(",", 9)[1:3] == ["0:00:00.10", "0:00:00.20"]
    assert ev[0].endswith("A{\\alpha&HFF&}BC") and ev[2].endswith("ABC")
    assert ev[2].split(",", 9)[2] == "0:00:00.90"


def test_replacement_and_empty_text():
    assert overlay_events(item(), "") == []
    doc = build_ass(tpl([item()]), {"text_1": "NEW"})
    assert doc.rstrip().endswith("NEW")


def test_caption_lines_and_karaoke():
    words = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in
             [("so", 0, 200), ("today", 200, 600), ("we", 700, 800), ("go", 2000, 2300)]]
    lines = caption_lines(words)
    assert [[w["w"] for w in l] for l in lines] == [["so", "today", "we"], ["go"]]
    doc = build_ass(tpl(words=words), {})
    dialogue = [l for l in doc.splitlines() if l.startswith("Dialogue:") and ",Caption," in l]
    assert len(dialogue) == 2
    assert "{\\k20}so {\\k50}today {\\k10}we" in dialogue[0]          # gap after "today" goes to its \k


def test_replace_words_spreads_over_speech_time():
    words = [{"w": "a", "start_ms": 0, "end_ms": 1000}, {"w": "b", "start_ms": 2000, "end_ms": 3000}]
    new = replace_words(words, "xx yy")
    assert new == [{"w": "xx", "start_ms": 0, "end_ms": 1000}, {"w": "yy", "start_ms": 2000, "end_ms": 3000}]
    assert replace_words(words, "") == []


def test_nothing_to_draw_is_none():
    assert build_ass(tpl(), {}) is None
```

Append to `tests/unit/test_slots.py`:
```python
def test_captions_key_is_kept(tmp_path):
    touch(tmp_path / "clip_1.mp4")
    (tmp_path / "texts.yaml").write_text("captions: new words here\ntext_1: HI\nother: x\n")
    media = resolve_media({"shots": [{"slot": "clip_1"}]}, tmp_path)
    assert media.texts == {"captions": "new words here", "text_1": "HI"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/user/.venv/bin/pytest tests/unit/test_ass.py tests/unit/test_slots.py -v`
Expected: FAIL — `No module named 'vtpl.render.ass'` and the captions-key assertion.

- [ ] **Step 3: Implement**

In `vtpl/render/slots.py` change the texts line to:
```python
    texts = {k: "" if v is None else str(v) for k, v in cfg.items()
             if str(k).startswith("text_") or k == "captions"}
```

`vtpl/render/ass.py`:
```python
"""Build an ASS subtitle document for text overlays and karaoke captions."""
from vtpl.fontlib import FALLBACK_ID, ass_font_size, entry

PLAY_W, PLAY_H = 1080, 1920
DEFAULT_CAPTION_STYLE = {"font": FALLBACK_ID, "size_px": 64, "color": "#FFFFFF",
                         "stroke": {"color": "#000000", "px": 4}, "shadow": False}
DEFAULT_CAPTION_BOX = {"x": 0.1, "y": 0.72, "w": 0.8, "h": 0.06}
LINE_WORDS, LINE_GAP_MS = 3, 600

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""
EVENTS = """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(ms: int) -> str:
    cs = max(0, round(ms / 10))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def style_line(name: str, st: dict, secondary_alpha: int = 0) -> str:
    e = entry(st["font"])
    stroke = st.get("stroke")
    outline, outline_col = (stroke["px"], ass_color(stroke["color"])) if stroke else (0, "&H00000000")
    fields = [name, e.family, ass_font_size(st["font"], st["size_px"]), ass_color(st["color"]),
              ass_color(st["color"], secondary_alpha), outline_col, "&H64000000", -1 if e.bold else 0,
              0, 0, 0, 100, 100, 0, 0, 1, outline, 3 if st["shadow"] else 0, 5, 0, 0, 0, 1]
    return "Style: " + ",".join(str(f) for f in fields)


def _centre(box: dict) -> tuple[int, int]:
    return round((box["x"] + box["w"] / 2) * PLAY_W), round((box["y"] + box["h"] / 2) * PLAY_H)


def _dialogue(start: int, end: int, style: str, tags: str, body: str) -> str:
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{{{tags}}}{body}"


def overlay_events(item: dict, text: str) -> list[str]:
    if text == "":
        return []
    start, end = item["start_ms"], item["end_ms"]
    dur = end - start
    a = item["anim"]
    in_ms, out_ms = min(a["in_ms"], dur // 2), min(a["out_ms"], dur // 2)
    cx, cy = _centre(item["box"])
    pos = f"\\pos({cx},{cy})"
    if a["in"] == "slide":
        off = a.get("in_from", {"dx": -0.3, "dy": 0.0})
        pos = f"\\move({cx + round(off['dx'] * PLAY_W)},{cy + round(off['dy'] * PLAY_H)},{cx},{cy},0,{in_ms})"
    elif a["out"] == "slide":
        off = a.get("out_to", {"dx": 0.3, "dy": 0.0})
        pos = f"\\move({cx},{cy},{cx + round(off['dx'] * PLAY_W)},{cy + round(off['dy'] * PLAY_H)},{dur - out_ms},{dur})"
    in_tags, out_tags = "", ""
    if a["in"] == "pop":
        k = round(in_ms * 2 / 3)
        in_tags = f"\\fscx80\\fscy80\\t(0,{k},\\fscx115\\fscy115)\\t({k},{in_ms},\\fscx100\\fscy100)"
    if a["out"] == "pop":
        out_tags = f"\\t({dur - out_ms},{dur},\\fscx0\\fscy0)"
    fade_in = in_ms if a["in"] == "fade" else 0
    fade_out = out_ms if a["out"] == "fade" else 0
    fad = f"\\fad({fade_in},{fade_out})" if fade_in or fade_out else ""
    name = item["id"]

    if a["in"] == "typewriter" and len(text) > 1:
        n = len(text)
        events = []
        for k in range(1, n + 1):
            s = start + round(in_ms * (k - 1) / n)
            e = start + round(in_ms * k / n) if k < n else end
            body = escape(text[:k]) + ("{\\alpha&HFF&}" + escape(text[k:]) if k < n else "")
            tags = f"\\an5{pos}" + (out_tags + (f"\\fad(0,{fade_out})" if fade_out else "") if k == n else "")
            events.append(_dialogue(s, e, name, tags, body))
        return events
    return [_dialogue(start, end, name, f"\\an5{pos}{fad}{in_tags}{out_tags}", escape(text))]


def caption_lines(words: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in words:
        if lines and len(lines[-1]) < LINE_WORDS and w["start_ms"] - lines[-1][-1]["end_ms"] <= LINE_GAP_MS:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def replace_words(words: list[dict], new_text: str) -> list[dict]:
    """Spread new words over the original speech time, in proportion to their length."""
    toks = new_text.split()
    if not toks or not words:
        return []
    slots = [(w["start_ms"], w["end_ms"]) for w in words]
    total = sum(e - s for s, e in slots)

    def at(frac: float, is_end: bool) -> int:
        """Map a share of total speech time to a clock time. A boundary between two slots maps to the
        end of the earlier slot for word ends and to the start of the later slot for word starts."""
        target, acc = frac * total, 0
        for s, e in slots:
            if acc + (e - s) > target or (is_end and acc + (e - s) >= target):
                return round(s + (target - acc))
            acc += e - s
        return slots[-1][1]

    lens = [len(t) for t in toks]
    total_chars, done, out = sum(lens), 0, []
    for tok, n in zip(toks, lens):
        out.append({"w": tok, "start_ms": at(done / total_chars, False),
                    "end_ms": at((done + n) / total_chars, True)})
        done += n
    return out


def caption_events(words: list[dict], box: dict) -> list[str]:
    cx, cy = _centre(box)
    events = []
    for line in caption_lines(words):
        parts = []
        for i, w in enumerate(line):
            nxt = line[i + 1]["start_ms"] if i + 1 < len(line) else w["end_ms"]
            parts.append(f"{{\\k{max(1, round((nxt - w['start_ms']) / 10))}}}{escape(w['w'])}")
        events.append(_dialogue(line[0]["start_ms"], line[-1]["end_ms"], "Caption", f"\\an5\\pos({cx},{cy})",
                                " ".join(parts)))
    return events


def build_ass(template: dict, texts: dict[str, str]) -> str | None:
    """The full ASS document, or None when there is nothing to draw."""
    styles, events = [], []
    for item in template["texts"]:
        if item["kind"] != "overlay":
            continue
        ev = overlay_events(item, texts.get(item["slot"], item["sample"]))
        if ev:
            styles.append(style_line(item["id"], item["style"]))
            events += ev
    words = template["captions"]["words"]
    if "captions" in texts:
        words = replace_words(words, texts["captions"])
    if words:
        ref = next((x for x in template["texts"] if x["id"] == template["captions"]["style_ref"]), None)
        styles.append(style_line("Caption", ref["style"] if ref else DEFAULT_CAPTION_STYLE, secondary_alpha=0x80))
        events += caption_events(words, ref["box"] if ref else DEFAULT_CAPTION_BOX)
    if not events:
        return None
    return HEADER.format(w=PLAY_W, h=PLAY_H) + "\n".join(styles) + "\n" + EVENTS + "\n".join(events) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/user/.venv/bin/pytest tests/unit/test_ass.py tests/unit/test_slots.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add vtpl/render/ass.py vtpl/render/slots.py tests/unit/test_ass.py tests/unit/test_slots.py
git commit -m "feat: build ASS overlays and karaoke captions"
```

---

### Task 13: Burn ASS into the render

**Files:**
- Modify: `vtpl/render/graph.py`, `vtpl/render/run.py`
- Test: `tests/unit/test_graph.py` (append), `tests/integration/test_render_text.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_graph.py`:
```python
def test_subtitles_filter_before_final_trim(tmp_path):
    from vtpl.render.graph import build_filter, plan_segments
    ass = tmp_path / "a b:c" / "out.ass"
    f = build_filter(plan_segments(T), False, 4000, ass=ass, fonts_dir=tmp_path / "fonts")
    tail = f.split(";")[-1]
    assert tail.startswith("[j2]subtitles=filename=") and "fontsdir=" in tail
    assert "a b\\\\:c" in tail                            # ':' -> '\:' (option level) -> '\\:' (graph level)
    assert tail.endswith("trim=duration=4.000,setpts=PTS-STARTPTS[vout]")
    assert "subtitles" not in build_filter(plan_segments(T), False, 4000)
```

`tests/integration/test_render_text.py`:
```python
import copy
import json

import pytest
from rapidfuzz import fuzz

from tests.synth.make_fixtures import solid_clip
from tests.unit.test_schema import GOOD, TEXT
from vtpl import frames
from vtpl.analyzers.text.detect import ocr
from vtpl.render.run import render

pytestmark = pytest.mark.slow


def _setup(tmp_path, texts_yaml=""):
    media = tmp_path / "mine"
    media.mkdir()
    solid_clip(media / "clip_1.mp4", "navy")
    solid_clip(media / "clip_2.mp4", "maroon")
    if texts_yaml:
        (media / "texts.yaml").write_text(texts_yaml)
    return media


def _read(out, frame_index):
    return frames.read_crops(out, {frame_index: [("f", (0, 0, 1080, 1920))]})["f"][frame_index]


def _texts(img):
    return [d["text"].upper() for d in ocr(img)]


def test_overlay_text_is_burned_with_replacement(tmp_path):
    t = copy.deepcopy(GOOD)
    item = copy.deepcopy(TEXT)
    item.update(start_ms=500, end_ms=2500, sample="HELLO", box={"x": 0.2, "y": 0.45, "w": 0.6, "h": 0.06})
    item["style"].update(size_px=140, stroke={"color": "#000000", "px": 6})
    t["texts"] = [item]
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(t))
    out = tmp_path / "out dir:1" / "out.mp4"
    render(tpl, _setup(tmp_path, "text_1: WORLD\n"), out)
    assert out.with_suffix(".ass").exists()
    assert any(fuzz.ratio(x, "WORLD") >= 80 for x in _texts(_read(out, 45)))       # 1.5 s
    assert not any(fuzz.ratio(x, "WORLD") >= 80 for x in _texts(_read(out, 6)))    # 0.2 s, before start


def test_captions_are_burned(tmp_path):
    t = copy.deepcopy(GOOD)
    t["captions"]["words"] = [{"w": "hello", "start_ms": 300, "end_ms": 900},
                              {"w": "there", "start_ms": 900, "end_ms": 1500}]
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(t))
    out = tmp_path / "out.mp4"
    render(tpl, _setup(tmp_path), out)
    assert any("HELLO" in x for x in _texts(_read(out, 36)))                       # 1.2 s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/user/.venv/bin/pytest tests/unit/test_graph.py tests/integration/test_render_text.py -v`
Expected: FAIL — `build_filter() got an unexpected keyword argument 'ass'`; integration tests find no text.

- [ ] **Step 3: Implement**

In `vtpl/render/graph.py` add:
```python
def _filter_path(p: Path) -> str:
    """Escape a path for an option value inside a filtergraph (two escaping levels)."""
    s = str(Path(p).resolve())
    for ch in "\\':":
        s = s.replace(ch, "\\" + ch)
    for ch in "\\'[],;":
        s = s.replace(ch, "\\" + ch)
    return s
```
Change `build_filter`'s signature and its final video line:
```python
def build_filter(segs: list[Segment], has_music: bool, duration_ms: int,
                 ass: Path | None = None, fonts_dir: Path | None = None) -> str:
```
```python
    subs = ""
    if ass is not None:
        subs = f"subtitles=filename={_filter_path(ass)}"
        if fonts_dir is not None:
            subs += f":fontsdir={_filter_path(fonts_dir)}"
        subs += ","
    parts.append(f"{cur}{subs}trim=duration={_s(duration_ms)},setpts=PTS-STARTPTS[vout]")
```
Change `build_args` to accept and pass them:
```python
def build_args(segs: list[Segment], clips: list[Path], trims_ms: list[int], music: Path | None,
               duration_ms: int, out: Path, ass: Path | None = None, fonts_dir: Path | None = None) -> list[str]:
```
and inside it:
```python
    args += ["-filter_complex", build_filter(segs, music is not None, duration_ms, ass, fonts_dir), "-map", "[vout]"]
```

In `vtpl/render/run.py` add imports `from vtpl import fontlib` and `from vtpl.render.ass import build_ass`, then replace the args/write block with:
```python
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ass_path = None
    ass_text = build_ass(template, media.texts)
    if ass_text is not None:
        ass_path = out.with_suffix(".ass")
        ass_path.write_text(ass_text, encoding="utf-8")
    args = build_args(segs, clips, trims, media.music, template["meta"]["duration_ms"], out,
                      ass=ass_path, fonts_dir=fontlib.fonts_dir())
    (out.parent / "render_cmd.txt").write_text("ffmpeg " + shlex.join(args) + "\n")
    run_ffmpeg(args)
    return media.warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/user/.venv/bin/pytest tests/unit/test_graph.py tests/integration/test_render_text.py tests/integration/test_render.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add vtpl/render/graph.py vtpl/render/run.py tests/unit/test_graph.py tests/integration/test_render_text.py
git commit -m "feat: burn text overlays and captions into renders"
```

---

### Task 14: Text round trip and README

**Files:**
- Create: `tests/roundtrip/test_roundtrip_text.py`
- Modify: `README.md`

- [ ] **Step 1: Write the round-trip test**

```python
import json

import numpy as np
import pytest
from rapidfuzz import fuzz

from tests.synth.make_fixtures import solid_clip
from vtpl.analyzers.text.style import delta_e, to_lab
from vtpl.pipeline import analyze
from vtpl.render.run import render

pytestmark = pytest.mark.slow
STAGES = {"ingest", "frames", "shots", "text", "merge"}


def lab(h):
    return to_lab(np.array([[int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)]], np.uint8))[0]


def overlays(t):
    return sorted((x for x in t["texts"] if x["kind"] == "overlay"), key=lambda x: x["start_ms"])


def test_text_roundtrip(tmp_path, text_video_basic):
    tpl_path = analyze(text_video_basic["path"], tmp_path / "work", only=STAGES)
    tpl = json.loads(tpl_path.read_text())
    media = tmp_path / "mine"
    media.mkdir()
    for shot in tpl["shots"]:
        solid_clip(media / f"{shot['slot']}.mp4", "darkslategray", seconds=5)
    out = tmp_path / "out.mp4"
    render(tpl_path, media, out)
    again = json.loads(analyze(out, tmp_path / "work2", only=STAGES).read_text())

    a, b = overlays(tpl), overlays(again)
    assert len(a) == len(b) == 3
    for x, y in zip(a, b):
        assert fuzz.ratio(x["sample"].upper(), y["sample"].upper()) >= 80
        assert abs(x["start_ms"] - y["start_ms"]) <= 100 and abs(x["end_ms"] - y["end_ms"]) <= 100
        for k in ("x", "y"):
            assert abs((x["box"][k] + x["box"]["w" if k == "x" else "h"] / 2)
                       - (y["box"][k] + y["box"]["w" if k == "x" else "h"] / 2)) <= 0.02
        assert delta_e(lab(x["style"]["color"]), lab(y["style"]["color"])) <= 15
        assert (x["style"]["stroke"] is None) == (y["style"]["stroke"] is None)
        assert x["anim"]["in"] == y["anim"]["in"]
        assert x["style"]["font"] == y["style"]["font"] or x["style"]["font"] == y["style"].get("font_guess")
```

- [ ] **Step 2: Run it**

Run: `/home/user/.venv/bin/pytest tests/roundtrip/test_roundtrip_text.py -v`
Expected: 1 passed. If a field differs, compare the libass render with the Pillow fixture for that text (save frames from both) — mismatches in size usually mean `ass_font_size` or `em_size_for`; in position, the `\an5\pos` centre; fix the renderer or analyser, not the tolerance.

- [ ] **Step 3: Update `README.md`**

Add after the `texts.yaml` example in "Use":
````markdown
Text overlays and captions found in the source are in `template.json` under `texts` and `captions`.
Replace them in `texts.yaml` (missing keys keep the original text; an empty value hides it):

```yaml
text_1: MY HEADLINE
text_2: ""            # hide this overlay
captions: new spoken line to show as karaoke captions
```

Fonts are matched against the bundled set in `vtpl/assets/fonts` (see its README for licences).
````

- [ ] **Step 4: Run the full suite**

Run: `/home/user/.venv/bin/pytest -q`
Expected: all pass (1 skipped: the Whisper test).

- [ ] **Step 5: Commit**

```bash
git add tests/roundtrip/test_roundtrip_text.py README.md
git commit -m "test: text overlay analyse-render-analyse round trip; document texts.yaml"
```
