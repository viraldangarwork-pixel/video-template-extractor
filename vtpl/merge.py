import copy
from pathlib import Path

from rapidfuzz import fuzz

from vtpl.analyzers.audio import beat_pattern, is_on_beat
from vtpl.fontlib import FALLBACK_ID
from vtpl.schema import validate
from vtpl.timeline import clamp_transitions
from vtpl.util.workdir import WorkDir

OUTPUT = "template.json"

FONT_MIN_CONFIDENCE = 0.5
CAPTION_SHARE = 0.6        # share of a text's words that must match nearby speech to count as a caption
CAPTION_SLACK_MS = 200


def _tokens(s: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in s.upper()).split()


def is_caption(track: dict, words: list[dict]) -> bool:
    toks = _tokens(track["text"])
    if not toks:
        return False
    near = [tok for w in words
            if w["end_ms"] >= track["start_ms"] - CAPTION_SLACK_MS and w["start_ms"] <= track["end_ms"] + CAPTION_SLACK_MS
            for tok in _tokens(w["w"])]
    hits = sum(1 for tok in toks if any(fuzz.ratio(tok, n) >= 80 for n in near))
    return hits / len(toks) >= CAPTION_SHARE


def _text_item(track: dict, item_id: str, slot: str, kind: str, dur: int) -> dict:
    font, st = track["font"], track["style"]
    stroke = {**st["stroke"], "color": st["stroke"]["color"].upper()} if st["stroke"] else None
    style_doc = {"font": font["id"], "font_confidence": font["confidence"], "size_px": st["size_px"],
                 "color": st["color"].upper(), "stroke": stroke, "shadow": st["shadow"]}
    if font["confidence"] < FONT_MIN_CONFIDENCE:
        style_doc["font"], style_doc["font_guess"] = FALLBACK_ID, font["id"]
    return {"id": item_id, "slot": slot, "sample": track["text"], "kind": kind,
            "start_ms": track["start_ms"], "end_ms": min(track["end_ms"], dur),
            "box": track["box"], "style": style_doc, "anim": track["anim"]}


def build_texts(tracks: list[dict], words: list[dict], dur: int) -> tuple[list[dict], str | None]:
    overlays, caption = [], None
    for tr in sorted(tracks, key=lambda t: t["start_ms"]):
        if tr["start_ms"] >= dur:
            continue
        if words and is_caption(tr, words):
            if caption is None:
                caption = _text_item(tr, "c1", "captions", "caption", dur)
            continue
        n = len(overlays) + 1
        overlays.append(_text_item(tr, f"t{n}", f"text_{n}", "overlay", dur))
    return overlays + ([caption] if caption else []), ("c1" if caption else None)


def _per_shot(doc: dict | None, n: int) -> list | None:
    """The doc's per-shot list when it matches the shot count, else None."""
    items = (doc or {}).get("shots")
    return items if isinstance(items, list) and len(items) == n else None


def build_template(meta: dict, shots_doc: dict, audio: dict | None, speech: dict | None, source_id: str,
                   text_doc: dict | None = None, motion_doc: dict | None = None,
                   grade_doc: dict | None = None) -> dict:
    dur = meta["duration_ms"]
    audio = audio or {"bpm": None, "beats_ms": [], "downbeat_hint_ms": []}
    audio = {**audio, **{k: [ms for ms in audio[k] if 0 <= ms <= dur] for k in ("beats_ms", "downbeat_hint_ms")}}
    words = [{"w": w["w"], "start_ms": w["start_ms"], "end_ms": min(w["end_ms"], dur)}
             for w in (speech or {}).get("words", []) if 0 <= w["start_ms"] < dur]
    texts, style_ref = build_texts((text_doc or {}).get("tracks", []), words, dur)
    shots = clamp_transitions(copy.deepcopy(shots_doc["shots"]))
    out_shots = [{
        "id": f"s{i + 1}",
        "start_ms": s["start_ms"],
        "end_ms": s["end_ms"],
        "slot": f"clip_{i + 1}",
        "on_beat": i > 0 and is_on_beat(s["start_ms"], audio["beats_ms"]),
        "transition_in": s["transition_in"],
    } for i, s in enumerate(shots)]
    for key, doc in (("motion", motion_doc), ("grade", grade_doc)):
        per = _per_shot(doc, len(out_shots))
        if per:
            for shot, value in zip(out_shots, per):
                shot[key] = value
    pattern = beat_pattern(out_shots, 60000 / audio["bpm"]) if audio["bpm"] else []
    template = {
        "version": 1,
        "meta": {"w": meta["w"], "h": meta["h"], "fps": meta["fps"], "duration_ms": meta["duration_ms"],
                 "bpm": audio["bpm"], "source_id": source_id},
        "audio": {"beats_ms": audio["beats_ms"], "downbeat_hint_ms": audio["downbeat_hint_ms"],
                  "beat_pattern": pattern, "music_slot": "music"},
        "shots": out_shots,
        "texts": texts,
        "captions": {"style_ref": style_ref, "words": words},
    }
    validate(template)
    return template


def run(wd: WorkDir, source: Path) -> list[str]:
    warnings = []
    optional = {}
    missing = {"audio.json": "no beats", "speech.json": "no captions", "text.json": "no text overlays",
               "motion.json": "no camera motion", "grade.json": "no colour grade"}
    for name, effect in missing.items():
        if wd.has(name):
            optional[name] = wd.read_json(name)
        else:
            warnings.append(f"{name} missing: {effect}")
    template = build_template(wd.read_json("meta.json"), wd.read_json("shots.json"),
                              optional.get("audio.json"), optional.get("speech.json"), source_id=wd.root.name,
                              text_doc=optional.get("text.json"), motion_doc=optional.get("motion.json"),
                              grade_doc=optional.get("grade.json"))
    wd.write_json(OUTPUT, template)
    return warnings
