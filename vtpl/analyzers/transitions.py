"""Pure transition classifiers. Frames: float32 (N, h, w, 3) in 0..1. Luma: (N,)."""
import numpy as np

DARK = 0.06          # luma below this is "black"
BRIGHT = 0.85        # luma above this is "white"
EPS = 1e-3
PAD = 3              # frames around a transition span treated as part of it


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _span(luma: np.ndarray, a: int, b: int, outward_brighter: bool) -> tuple[int, int]:
    """Grow [a, b) outwards while luma keeps moving away from the dip/peak. Inclusive result."""
    s, e = a, b - 1
    sign = 1 if outward_brighter else -1
    while s > 0 and sign * (luma[s - 1] - luma[s]) > EPS:
        s -= 1
    while e < len(luma) - 1 and sign * (luma[e + 1] - luma[e]) > EPS:
        e += 1
    return s, e


def _boundary(kind: str, s: int, e: int) -> dict:
    return {"frame": (s + e + 1) // 2, "type": kind, "duration_frames": e - s + 1, "span": (s, e)}


def find_fades(luma: np.ndarray, min_ramp: int = 3) -> list[dict]:
    out = []
    for a, b in runs(luma < DARK):
        if a == 0 or b == len(luma):
            continue                                  # starts or ends in black: not between shots
        s, e = _span(luma, a, b, outward_brighter=True)
        if (a - s) >= min_ramp or (e - (b - 1)) >= min_ramp:
            out.append(_boundary("fade", s, e))
    return out


def find_flashes(luma: np.ndarray, frames: np.ndarray, max_len: int = 4,
                 min_rise: float = 0.3, min_change: float = 0.1) -> list[dict]:
    out = []
    for a, b in runs(luma > BRIGHT):
        if b - a > max_len or a < PAD or b + PAD > len(luma):
            continue
        peak = float(luma[a:b].max())
        if luma[a - PAD] > peak - min_rise or luma[b + PAD - 1] > peak - min_rise:
            continue
        if float(np.abs(frames[a - PAD] - frames[b + PAD - 1]).mean()) < min_change:
            continue                                  # same picture before and after: in-shot flash
        s, e = _span(luma, a, b, outward_brighter=False)
        out.append(_boundary("flash", s, e))
    return out


def dissolve_span(frames: np.ndarray, c: int, max_half: int = 12, res_thr: float = 0.25,
                  min_blend: int = 3) -> tuple[int, int] | None:
    """Inclusive frame span of a linear blend around frame c, or None."""
    best = None
    for half in range(2, max_half + 1):
        a, b = c - half - 1, c + half
        if a < 0 or b >= len(frames):
            break
        A, B = frames[a], frames[b]
        d = A - B
        denom = float((d * d).sum())
        if denom < 1e-6:
            break
        alphas, ok = [], True
        for i in range(a + 1, b):
            F = frames[i]
            al = float(((F - B) * d).sum() / denom)
            res = float(np.linalg.norm(F - (al * A + (1 - al) * B)) / np.sqrt(denom))
            if res > res_thr:
                ok = False
                break
            alphas.append(al)
        if not ok:
            break
        monotonic = all(x >= y - 0.05 for x, y in zip(alphas, alphas[1:]))
        idx = [a + 1 + j for j, al in enumerate(alphas) if 0.05 < al < 0.95]
        if monotonic and len(idx) >= min_blend:
            best = (idx[0], idx[-1])
    return best


def _dissolve(frames: np.ndarray, c: int) -> tuple[int, int] | None:
    """dissolve_span, re-fitted around the span's midpoint: detectors often flag a dissolve's
    first frame, and a window centred there truncates the far end of the blend."""
    span = dissolve_span(frames, c)
    if span is None:
        return None
    again = dissolve_span(frames, (span[0] + span[1] + 1) // 2)
    return again if again and again[1] - again[0] > span[1] - span[0] else span


def long_range_diff(frames: np.ndarray, k: int = 6) -> np.ndarray:
    L = np.zeros(len(frames), dtype=np.float32)
    for i in range(k, len(frames) - k):
        L[i] = float(np.abs(frames[i - k] - frames[i + k]).mean())
    return L


def dissolve_candidates(frames: np.ndarray, k: int = 6, thr: float = 0.15) -> list[int]:
    L = long_range_diff(frames, k)
    return [i for i in range(1, len(L) - 1) if L[i] >= thr and L[i] >= L[i - 1] and L[i] > L[i + 1]]


def classify_boundaries(cut_frames: list[int], frames: np.ndarray, luma: np.ndarray) -> list[dict]:
    """Merge detector cuts with luma/blend analysis into one sorted boundary list."""
    out = find_fades(luma) + find_flashes(luma, frames)

    def covered(f: int) -> bool:
        return any(x["span"][0] - PAD <= f <= x["span"][1] + PAD for x in out)

    for c in sorted(set(cut_frames)):
        if covered(c):
            continue
        span = _dissolve(frames, c)
        out.append(_boundary("dissolve", *span) if span else
                   {"frame": c, "type": "cut", "duration_frames": 0, "span": (c, c)})
    for c in dissolve_candidates(frames):
        if covered(c) or any(abs(c - x["frame"]) <= 6 for x in out):
            continue
        span = _dissolve(frames, c)
        if span:
            out.append(_boundary("dissolve", *span))
    return sorted(out, key=lambda x: x["frame"])
