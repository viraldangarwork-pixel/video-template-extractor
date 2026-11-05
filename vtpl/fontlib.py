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
