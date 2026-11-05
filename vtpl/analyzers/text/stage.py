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


def start_window_ms(t: dict) -> int:
    """How far before the first OCR sample the start is searched. A typewriter reveal (the first
    sample reads only a prefix) began earlier than OCR could read it: a lone first letter is
    often not detected at all, so the search reaches back a further animation length."""
    first = " ".join(t["obs"][0]["text"].upper().split())
    if first != " ".join(t["text"].upper().split()) and track.is_prefix(first, t["text"]):
        return WINDOW_MS + ANIM_MS
    return WINDOW_MS


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
        start_ids = list(range(max(0, s0 - ms_to_frame(start_window_ms(t), fps)), min(last, s0 + an) + 1))
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
