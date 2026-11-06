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


def _ink_columns(mask: np.ndarray) -> np.ndarray:
    """The mask without its empty columns: letter and word spacing removed."""
    return mask[:, mask.any(axis=0)]


def rank_fonts(text: str, fill_mask: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
    """Catalog fonts by glyph similarity to the mask. Each font scores the better of the text as read
    and a comparison with the spacing removed from both sides, so that OCR merging or splitting
    words ("LINKIN BIO") does not decide the font."""
    target = crop_to_ink(fill_mask)
    if target.size == 0:
        raise ValueError("empty text mask")
    packed, squeezed = _ink_columns(target), "".join(text.split())
    scored = []
    for e in catalog():
        as_read = similarity(target, ink_mask(text, e.id, target.shape[0]))
        unspaced = similarity(packed, _ink_columns(ink_mask(squeezed, e.id, target.shape[0]))) if squeezed else 0.0
        scored.append((e.id, max(as_read, unspaced)))
    return sorted(scored, key=lambda s: s[1], reverse=True)[:top]


def em_size_for(text: str, font_id: str, ink_h: int) -> int:
    """Pillow em size (px) at which `text` in this font has the given ink height."""
    return max(1, round(RENDER_SIZE * ink_h / _glyphs(text, font_id).shape[0]))
