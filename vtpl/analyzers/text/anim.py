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
