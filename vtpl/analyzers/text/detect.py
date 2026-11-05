"""OCR on every SAMPLE_EVERY-th cached frame (RapidOCR, CPU)."""
from functools import lru_cache

import cv2
import numpy as np

from vtpl import frames as fr
from vtpl.util.workdir import WorkDir

SAMPLE_EVERY = 2          # every 2nd cached frame: one sample per ~133 ms at 15 fps
MIN_SCORE = 0.6
SMALL_BOX_PX = 20         # boxes shorter than this in the 540x960 cache are re-read at full resolution
PAD = 60                  # replicate-border padding: the detector splits lines that touch the image edge
SMALL_MIN_SCORE = 0.3     # keep weak small detections until the full-resolution re-read


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr(img_bgr: np.ndarray) -> list[dict]:
    """[{box: [x, y, w, h] int px, text, score}] for one BGR image."""
    padded = cv2.copyMakeBorder(img_bgr, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    result, _ = _engine()(padded)
    out = []
    for quad, text, score in result or []:
        xs, ys = [p[0] - PAD for p in quad], [p[1] - PAD for p in quad]
        x0, y0 = max(0, min(xs)), max(0, min(ys))
        text = text.strip()
        if text:
            out.append({"box": [round(x0), round(y0), round(max(xs) - x0), round(max(ys) - y0)],
                        "text": text, "score": float(score)})
    return out


def to_norm(box_px: list[int], w: int, h: int) -> list[float]:
    return [box_px[0] / w, box_px[1] / h, box_px[2] / w, box_px[3] / h]


def detect_samples(wd: WorkDir) -> list[dict]:
    """[{t_ms, frame (full-rate index), dets: [{box (normalised), text, score}]}] in time order."""
    cache = wd.read_json(fr.OUTPUT)
    ratio = round(wd.read_json("meta.json")["fps"] / cache["fps"])
    samples, small = [], []
    for ci in range(0, len(cache["files"]), SAMPLE_EVERY):
        dets = []
        for d in ocr(fr.cached_frame(wd, ci)):
            is_small = d["box"][3] < SMALL_BOX_PX
            if d["score"] < (SMALL_MIN_SCORE if is_small else MIN_SCORE):
                continue
            dets.append({"box": to_norm(d["box"], cache["w"], cache["h"]), "text": d["text"],
                         "score": round(d["score"], 3)})
            if is_small:
                small.append((len(samples), len(dets) - 1))
        samples.append({"t_ms": fr.cache_time_ms(ci, cache["fps"]), "frame": ci * ratio, "dets": dets})
    _reread_small(wd, samples, small)
    for s in samples:
        s["dets"] = [d for d in s["dets"] if d["score"] >= MIN_SCORE]
    return samples


def _reread_small(wd: WorkDir, samples: list[dict], small: list[tuple[int, int]]) -> None:
    if not small:
        return
    meta = wd.read_json("meta.json")
    requests: dict[int, list] = {}
    for si, di in small:
        x, y, w, h = samples[si]["dets"][di]["box"]
        px = (round(x * meta["w"]), round(y * meta["h"]), round(w * meta["w"]), round(h * meta["h"]))
        pad = px[3] // 2 + 4
        rect = (px[0] - pad, px[1] - pad, px[2] + 2 * pad, px[3] + 2 * pad)
        requests.setdefault(samples[si]["frame"], []).append((f"{si}:{di}", rect))
    crops = fr.read_crops(wd.path("norm.mp4"), requests)
    for si, di in small:
        frame = samples[si]["frame"]
        crop = crops.get(f"{si}:{di}", {}).get(frame)
        if crop is None or crop.size == 0:
            continue
        found = ocr(crop)
        if found:
            best = max(found, key=lambda d: d["score"])
            det = samples[si]["dets"][di]
            if best["score"] > det["score"]:
                det["text"], det["score"] = best["text"], round(best["score"], 3)
