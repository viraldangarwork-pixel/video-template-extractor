import copy

import pytest

from vtpl.schema import TemplateError, validate
from vtpl.timeline import clamp_transitions

GOOD = {
    "version": 1,
    "meta": {"w": 1080, "h": 1920, "fps": 30, "duration_ms": 3000, "bpm": 120.0, "source_id": "abc123abc123"},
    "audio": {"beats_ms": [500, 1000], "downbeat_hint_ms": [500], "beat_pattern": [2, 4], "music_slot": "music"},
    "shots": [
        {"id": "s1", "start_ms": 0, "end_ms": 1000, "slot": "clip_1", "on_beat": False,
         "transition_in": {"type": "cut", "duration_ms": 0}},
        {"id": "s2", "start_ms": 1000, "end_ms": 3000, "slot": "clip_2", "on_beat": True,
         "transition_in": {"type": "dissolve", "duration_ms": 400}},
    ],
    "texts": [],
    "captions": {"style_ref": None, "words": [{"w": "hi", "start_ms": 100, "end_ms": 300}]},
}


def test_good_template_passes():
    validate(copy.deepcopy(GOOD))


def test_schema_error_names_path():
    t = copy.deepcopy(GOOD)
    t["shots"][1]["transition_in"]["type"] = "wipe"
    with pytest.raises(TemplateError, match="shots/1/transition_in/type"):
        validate(t)


@pytest.mark.parametrize("mutate,msg", [
    (lambda t: t["shots"][0].update(start_ms=10), "shots\\[0\\].start_ms must be 0"),
    (lambda t: t["shots"][1].update(start_ms=1100), "must equal shots\\[0\\].end_ms"),
    (lambda t: t["meta"].update(duration_ms=2500), "last shot must end at meta.duration_ms"),
    (lambda t: t["shots"][1]["transition_in"].update(duration_ms=1000), "must be shorter than both"),
])
def test_timeline_errors(mutate, msg):
    t = copy.deepcopy(GOOD)
    mutate(t)
    with pytest.raises(TemplateError, match=msg):
        validate(t)


def test_clamp_transitions():
    shots = copy.deepcopy(GOOD["shots"])
    shots[1]["transition_in"]["duration_ms"] = 900
    clamp_transitions(shots)
    assert shots[1]["transition_in"] == {"type": "dissolve", "duration_ms": 500}
    shots[1]["transition_in"] = {"type": "fade", "duration_ms": 0}
    clamp_transitions(shots)
    assert shots[1]["transition_in"] == {"type": "cut", "duration_ms": 0}
