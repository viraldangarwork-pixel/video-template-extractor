"""Find the exact first/last frame of a text track from full-rate crops around an OCR sample."""
import numpy as np

FRAC = 0.15        # a frame has text once it is 15 % of the way from the empty crop to the text crop
MASK_SHARE = 0.5   # text pixels: those that change at least half as much as the most-changed ones


def text_mask(text_ref: np.ndarray, empty_ref: np.ndarray) -> np.ndarray | None:
    """The pixels the text changes most between the empty and the text crop (None if none change).
    Background motion changes pixels far less than text drawn over them does, so it stays out."""
    d = np.linalg.norm(text_ref.astype(np.float32) - empty_ref.astype(np.float32), axis=2)
    top = float(np.percentile(d, 99))
    return d >= MASK_SHARE * top if top > 1e-6 else None


def text_level(crop: np.ndarray, text_ref: np.ndarray, empty_ref: np.ndarray, mask: np.ndarray | None) -> float:
    """How far `crop` has moved from the empty crop towards the text crop on the text's own pixels
    (least squares): 0 = empty, 1 = like the text crop. 1.0 when the references do not differ."""
    if mask is None or not mask.any():
        return 1.0
    v = text_ref[mask].astype(np.float32) - empty_ref[mask].astype(np.float32)
    u = crop[mask].astype(np.float32) - empty_ref[mask].astype(np.float32)
    return float((u * v).sum() / max(float((v * v).sum()), 1e-6))


def text_flags(crops: list[np.ndarray], text_ref: np.ndarray, empty_ref: np.ndarray, frac: float = FRAC) -> list[bool]:
    mask = text_mask(text_ref, empty_ref)
    if mask is None:
        return [True] * len(crops)
    return [text_level(c, text_ref, empty_ref, mask) >= frac for c in crops]


def fill_flags(counts: list[int], text_count: int, frac: float = FRAC) -> list[bool]:
    """Frames showing at least `frac` of the text crop's fill-colour pixels: robust to background
    motion and catches text away from its final place (a slide), but not the faint start of a fade."""
    return [text_count > 0 and n >= frac * text_count for n in counts]


def _merge(flags: list[bool], also: list[bool] | None) -> list[bool]:
    return flags if also is None else [a or b for a, b in zip(flags, also)]


def refine_start(ids: list[int], crops: list[np.ndarray], confirmed: int,
                 text_ref: np.ndarray, empty_ref: np.ndarray, also: list[bool] | None = None) -> int:
    """First frame of the run of text frames that ends at ids[confirmed]. `also` flags extra frames
    as showing text (e.g. fill_flags)."""
    flags = _merge(text_flags(crops, text_ref, empty_ref), also)
    j = confirmed
    if not flags[j]:
        return ids[j]
    while j > 0 and flags[j - 1]:
        j -= 1
    return ids[j]


def refine_end(ids: list[int], crops: list[np.ndarray], confirmed: int,
               text_ref: np.ndarray, empty_ref: np.ndarray, also: list[bool] | None = None) -> int:
    """Last frame of the run of text frames that starts at ids[confirmed]."""
    flags = _merge(text_flags(crops, text_ref, empty_ref), also)
    j = confirmed
    if not flags[j]:
        return ids[j]
    while j < len(ids) - 1 and flags[j + 1]:
        j += 1
    return ids[j]
