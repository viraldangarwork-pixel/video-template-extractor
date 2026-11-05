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
