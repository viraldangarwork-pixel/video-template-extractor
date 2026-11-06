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
BOX_PAD = 0.25        # timing / contrast crops: the box extended by this share of its height
EDGE_FILL_SHARE = 0.5 # a first / last frame with this share of the text's fill pixels shows the text
BAND_SCALE = 0.5      # band crops (slide geometry, fill counts) are kept at this scale to save memory
MAX_TRACKS_PER_PASS = 8   # crops of at most this many tracks are held at once (one video pass each)


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


def plan_batches(spans: dict[str, tuple[int, int]], max_tracks: int = MAX_TRACKS_PER_PASS) -> list[list[str]]:
    """Group tracks, in order of their first needed frame, into batches of at most `max_tracks`;
    each batch is one read_crops pass, so memory is bounded by the batch, not the whole video."""
    order = sorted(spans, key=lambda k: spans[k])
    return [order[i:i + max_tracks] for i in range(0, len(order), max_tracks)]


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
    plans, track_requests = {}, {}
    for t in tracks:
        r = _px(t["box"], W, H)
        s0, s1 = t["obs"][0]["frame"], t["obs"][-1]["frame"]
        start_ids = list(range(max(0, s0 - ms_to_frame(start_window_ms(t), fps)), min(last, s0 + an) + 1))
        end_ids = list(range(max(0, s1 - an), min(last, s1 + win) + 1))
        style_ids = _style_frames(wd, t, cache, ratio)
        req = []
        for f in sorted(set(start_ids) | set(end_ids)):
            req.append((f, (t["id"] + ":band", _band(r, W), BAND_SCALE)))
            req.append((f, (t["id"] + ":box", _expand(r, BOX_PAD))))
        for f in style_ids:
            req.append((f, (t["id"] + ":style", _expand(r, STYLE_PAD))))
        plans[t["id"]] = (start_ids, end_ids, style_ids, s0, s1, last)
        track_requests[t["id"]] = req

    results, warnings = {}, []
    spans = {k: (min(f for f, _ in req), max(f for f, _ in req)) for k, req in track_requests.items()}
    by_id = {t["id"]: t for t in tracks}
    for batch in plan_batches(spans):
        requests: dict[int, list] = defaultdict(list)
        for k in batch:
            for f, rq in track_requests[k]:
                requests[f].append(rq)
        crops = fr.read_crops(wd.path("norm.mp4"), dict(requests))
        for k in batch:
            t = by_id[k]
            try:
                results[k] = _analyse(t, plans[k], crops, meta)
            except Exception as e:
                warnings.append(f"{k} ({t['text']!r}) skipped: {e}")
        del crops
    out = [results[t["id"]] for t in tracks if t["id"] in results]
    wd.write_json(OUTPUT, {"tracks": out})
    return warnings


def _shows_text(crop, text_crop, fill_bgr) -> bool:
    """Whether `crop` shows the text about as fully as `text_crop` does, by fill-colour pixel count."""
    n_text = int(anim.fill_mask(text_crop, fill_bgr).sum())
    return n_text > 0 and anim.fill_mask(crop, fill_bgr).sum() >= EDGE_FILL_SHARE * n_text


def _analyse(t: dict, plan, crops: dict, meta: dict) -> dict:
    start_ids, end_ids, style_ids, s0, s1, last = plan
    W, H, fps = meta["w"], meta["h"], meta["fps"]
    styles = [style.analyze_style(crops[t["id"] + ":style"][f]) for f in style_ids]
    best_style = styles[0]                                   # style_ids are sharpest first
    fill = best_style.color_bgr

    # Timing from the text's own pixels in its box: over a full-width band, background motion
    # crosses the threshold.
    band, box = crops[t["id"] + ":band"], crops[t["id"] + ":box"]
    start_box, end_box = [box[f] for f in start_ids], [box[f] for f in end_ids]
    # Fill-colour pixels in the band also count: they find text away from its box (a slide).
    counts = {f: int(anim.fill_mask(band[f], fill).sum()) for f in set(start_ids) | set(end_ids)}
    bw, bh = W * BAND_SCALE, H * BAND_SCALE                  # band geometry is in downscaled pixels
    if start_ids[0] == 0 and _shows_text(box[0], box[s0], fill):
        f_start = 0                                          # on screen from the first frame: no empty frame to compare
    else:
        f_start = refine.refine_start(start_ids, start_box, start_ids.index(s0), box[s0], start_box[0],
                                      also=refine.fill_flags([counts[f] for f in start_ids], counts[s0]))
    if end_ids[-1] == last and _shows_text(box[last], box[s1], fill):
        f_end = last
    else:
        f_end = refine.refine_end(end_ids, end_box, end_ids.index(s1), box[s1], end_box[-1],
                                  also=refine.fill_flags([counts[f] for f in end_ids], counts[s1]))

    ranked = font.rank_fonts(t["text"], best_style.fill_mask)
    font_id, confidence = ranked[0]
    size_px = font.em_size_for(t["text"], font_id, style.ink_height(best_style.fill_mask))

    an, step = ms_to_frame(ANIM_MS, fps), 1000 / fps
    in_ids = [f for f in start_ids if f_start <= f <= f_start + an]
    out_ids = [f for f in reversed(end_ids) if f_end - an <= f <= f_end]
    none = {"kind": "none", "ms": 0}
    # A text already on screen at the first frame (or still on at the last) shows no animation there.
    a_in = none if f_start == 0 else anim.classify(anim.measure(
        [band[f] for f in in_ids], fill, band[start_ids[0]], [box[f] for f in in_ids], start_box[0]), bw, bh, step)
    a_out = none if f_end == last else anim.classify(anim.measure(
        [band[f] for f in out_ids], fill, band[end_ids[-1]], [box[f] for f in out_ids], end_box[-1]), bw, bh, step)
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
