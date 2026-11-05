import math

from jsonschema import Draft202012Validator

INT = {"type": "integer", "minimum": 0}
MS_LIST = {"type": "array", "items": INT}

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
        "texts": {"type": "array", "items": {"type": "object"}},
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
            },
        },
    },
}

_VALIDATOR = Draft202012Validator(TEMPLATE_SCHEMA)


class TemplateError(ValueError):
    pass


def check_timeline(t: dict) -> None:
    shots, dur, fps = t["shots"], t["meta"]["duration_ms"], t["meta"]["fps"]
    frame_ms = math.ceil(1000 / fps)
    if shots[0]["start_ms"] != 0:
        raise TemplateError("shots[0].start_ms must be 0")
    for i, s in enumerate(shots):
        if s["end_ms"] <= s["start_ms"]:
            raise TemplateError(f"shots[{i}] has end_ms <= start_ms")
        if s["end_ms"] - s["start_ms"] < frame_ms:
            raise TemplateError(f"shots[{i}] is {s['end_ms'] - s['start_ms']} ms long, shorter than one frame "
                                f"({frame_ms} ms at {fps:g} fps)")
        if i and s["start_ms"] != shots[i - 1]["end_ms"]:
            raise TemplateError(f"shots[{i}].start_ms must equal shots[{i - 1}].end_ms")
        d = s["transition_in"]["duration_ms"]
        if i and d and d >= min(s["end_ms"] - s["start_ms"], shots[i - 1]["end_ms"] - shots[i - 1]["start_ms"]):
            raise TemplateError(f"shots[{i}].transition_in.duration_ms ({d}) must be shorter than both neighbouring shots")
    if shots[-1]["end_ms"] != dur:
        raise TemplateError(f"last shot must end at meta.duration_ms ({dur})")


def validate(t: dict) -> None:
    errors = sorted(_VALIDATOR.iter_errors(t), key=lambda e: list(e.absolute_path))
    if errors:
        e = errors[0]
        loc = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise TemplateError(f"template.json invalid at {loc}: {e.message}")
    check_timeline(t)
