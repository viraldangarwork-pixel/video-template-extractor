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
