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
    return round(inter / union, 9) if union > 0 else 0.0


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
    longest = max(counts, key=len)
    if all(is_prefix(t, longest) for t in counts):     # typewriter: a partial prefix may win the vote
        text = longest
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
