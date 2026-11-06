
from jsonschema import Draft202012Validator

from vtpl import fontlib

INT = {"type": "integer", "minimum": 0}
MS_LIST = {"type": "array", "items": INT}

ANIMS = ["none", "pop", "fade", "slide", "typewriter"]
UNIT = {"type": "number", "minimum": 0, "maximum": 1}
OFFSET = {"type": "object", "required": ["dx", "dy"], "properties": {"dx": {"type": "number"}, "dy": {"type": "number"}}}
SCALE = {"type": "number", "minimum": 0.5, "maximum": 2}      # camera zoom factor
SHIFT = {"type": "number", "minimum": -0.5, "maximum": 0.5}   # camera pan, share of the frame
TEXT_ITEM = {
    "type": "object",
    "required": ["id", "slot", "sample", "kind", "start_ms", "end_ms", "box", "style", "anim"],
    "properties": {
        "id": {"type": "string"}, "slot": {"type": "string"}, "sample": {"type": "string"},
        "kind": {"enum": ["overlay", "caption"]}, "start_ms": INT, "end_ms": INT,
        "box": {"type": "object", "required": ["x", "y", "w", "h"],
                "properties": {"x": UNIT, "y": UNIT, "w": UNIT, "h": UNIT}},
        "style": {
            "type": "object",
            "required": ["font", "font_confidence", "size_px", "color", "stroke", "shadow"],
            "properties": {
                "font": {"type": "string"}, "font_guess": {"type": "string"},
                "font_confidence": {"type": "number"}, "size_px": {"type": "integer", "minimum": 1},
                "color": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                "stroke": {"oneOf": [{"type": "null"}, {
                    "type": "object", "required": ["color", "px"],
                    "properties": {"color": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                                   "px": {"type": "integer", "minimum": 0}}}]},
                "shadow": {"type": "boolean"},
            },
        },
        "anim": {"type": "object", "required": ["in", "in_ms", "out", "out_ms"],
                 "properties": {"in": {"enum": ANIMS}, "out": {"enum": ANIMS}, "in_ms": INT, "out_ms": INT,
                                "in_from": OFFSET, "out_to": OFFSET}},
    },
}

TEMPLATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["version", "meta", "audio", "shots", "texts", "captions"],
    "properties": {
        "version": {"const": 1},
        "meta": {
            "type": "object",
            "required": ["w", "h", "fps", "duration_ms", "bpm", "source_id"],
            "properties": {
                "w": {"type": "integer", "minimum": 1}, "h": {"type": "integer", "minimum": 1},
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "duration_ms": {"type": "integer", "minimum": 1},
                "bpm": {"type": ["number", "null"]},
                "source_id": {"type": "string"},
            },
        },
        "audio": {
            "type": "object",
            "required": ["beats_ms", "downbeat_hint_ms", "beat_pattern", "music_slot"],
            "properties": {
                "beats_ms": MS_LIST, "downbeat_hint_ms": MS_LIST,
                "beat_pattern": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "music_slot": {"type": "string"},
            },
        },
        "shots": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/shot"}},
        "texts": {"type": "array", "items": TEXT_ITEM},
        "captions": {
            "type": "object",
            "required": ["style_ref", "words"],
            "properties": {
                "style_ref": {"type": ["string", "null"]},
                "words": {"type": "array", "items": {
                    "type": "object", "required": ["w", "start_ms", "end_ms"],
                    "properties": {"w": {"type": "string"}, "start_ms": INT, "end_ms": INT}}},
            },
        },
    },
    "$defs": {
        "shot": {
            "type": "object",
            "required": ["id", "start_ms", "end_ms", "slot", "on_beat", "transition_in"],
            "properties": {
                "id": {"type": "string"}, "start_ms": INT, "end_ms": INT, "slot": {"type": "string"},
                "on_beat": {"type": "boolean"},
                "transition_in": {
                    "type": "object", "required": ["type", "duration_ms"],
                    "properties": {"type": {"enum": ["cut", "fade", "dissolve", "flash"]}, "duration_ms": INT},
                },
                "motion": {
                    "type": "object", "required": ["type", "scale_from", "scale_to", "dx", "dy"],
                    "properties": {"type": {"enum": ["static", "zoom_in", "zoom_out", "pan"]},
                                   "scale_from": SCALE, "scale_to": SCALE, "dx": SHIFT, "dy": SHIFT},
                },
                "grade": {
                    "type": "object", "required": ["luma_mean", "luma_std", "chroma"],
                    "properties": {"luma_mean": UNIT, "luma_std": UNIT, "chroma": UNIT},
                },
            },
        },
    },
}

_VALIDATOR = Draft202012Validator(TEMPLATE_SCHEMA)


class TemplateError(ValueError):
    pass


def check_timeline(t: dict) -> None:
    shots, dur, fps = t["shots"], t["meta"]["duration_ms"], t["meta"]["fps"]
    if shots[0]["start_ms"] != 0:
        raise TemplateError("shots[0].start_ms must be 0")
    for i, s in enumerate(shots):
        if s["end_ms"] <= s["start_ms"]:
            raise TemplateError(f"shots[{i}] has end_ms <= start_ms")
        if round(s["end_ms"] * fps / 1000) <= round(s["start_ms"] * fps / 1000):     # the renderer's frame mapping
            raise TemplateError(f"shots[{i}] covers no whole frame at {fps:g} fps")
        if i and s["start_ms"] != shots[i - 1]["end_ms"]:
            raise TemplateError(f"shots[{i}].start_ms must equal shots[{i - 1}].end_ms")
        d = s["transition_in"]["duration_ms"]
        if i and d and d >= min(s["end_ms"] - s["start_ms"], shots[i - 1]["end_ms"] - shots[i - 1]["start_ms"]):
            raise TemplateError(f"shots[{i}].transition_in.duration_ms ({d}) must be shorter than both neighbouring shots")
    if shots[-1]["end_ms"] != dur:
        raise TemplateError(f"last shot must end at meta.duration_ms ({dur})")
    for i, x in enumerate(t["texts"]):
        if not 0 <= x["start_ms"] < x["end_ms"] <= dur:
            raise TemplateError(f"texts[{i}] must lie within 0..{dur} ms with start before end")


def validate(t: dict) -> None:
    errors = sorted(_VALIDATOR.iter_errors(t), key=lambda e: list(e.absolute_path))
    if errors:
        e = errors[0]
        loc = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise TemplateError(f"template.json invalid at {loc}: {e.message}")
    check_fonts(t)
    check_style_ref(t)
    check_timeline(t)


def check_style_ref(t: dict) -> None:
    """captions.style_ref, when set, must name one of the texts (the renderer styles captions after it)."""
    ref = t["captions"]["style_ref"]
    if ref is not None and ref not in {x["id"] for x in t["texts"]}:
        raise TemplateError(f"captions/style_ref: {ref!r} is not the id of any item in texts")


def check_fonts(t: dict) -> None:
    """Every text's font must be in the bundled catalog (the renderer looks it up by id)."""
    known = {e.id for e in fontlib.catalog()}
    for i, x in enumerate(t["texts"]):
        font = x["style"]["font"]
        if font not in known:
            raise TemplateError(f"texts/{i}/style/font: unknown font {font!r}; "
                                f"see {fontlib.fonts_dir() / 'catalog.json'}")
