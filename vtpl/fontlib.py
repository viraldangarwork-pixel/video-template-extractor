"""Bundled font catalog: lookup, Pillow fonts and ASS size conversion."""
import json
import os
import struct
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


def _tables(data: bytes) -> dict[bytes, bytes]:
    n = struct.unpack_from(">H", data, 4)[0]
    out = {}
    for i in range(n):
        tag, _, off, length = struct.unpack_from(">4sIII", data, 12 + 16 * i)
        out[tag] = data[off:off + length]
    return out


def line_height_em(font_id: str) -> float:
    """The font's line height in em, as libass computes it: like GDI it takes OS/2 usWinAscent +
    usWinDescent, falling back to the OS/2 typo metrics and then to hhea when those sum to zero."""
    return _line_height_em(str(font_path(entry(font_id))))


@lru_cache(maxsize=256)
def _line_height_em(path: str) -> float:
    t = _tables(Path(path).read_bytes())
    upm = struct.unpack_from(">H", t[b"head"], 18)[0]
    os2 = t.get(b"OS/2", b"")
    if len(os2) >= 78:
        win_asc, win_desc = struct.unpack_from(">hh", os2, 74)    # read signed, as libass does
        if win_asc + win_desc != 0:
            return (win_asc + win_desc) / upm
        typo_asc, typo_desc = struct.unpack_from(">hh", os2, 68)
        if typo_asc - typo_desc != 0:
            return (typo_asc - typo_desc) / upm
    asc, desc = struct.unpack_from(">hh", t[b"hhea"], 4)
    return (asc - desc) / upm


def ass_font_size(font_id: str, em_px: int) -> int:
    """libass sizes a font so that its (win) line height equals the ASS Fontsize, so convert from em."""
    return round(em_px * line_height_em(font_id))
