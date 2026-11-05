import copy

import pytest

from vtpl.render.snap import snap_template
from vtpl.schema import TemplateError, validate
from tests.unit.test_schema import GOOD


def test_snap_rescales_shots_and_words():
    t = snap_template(copy.deepcopy(GOOD), new_bpm=60.0)       # pattern [2, 4] at 1000 ms/beat
    validate(t)
    assert [(s["start_ms"], s["end_ms"]) for s in t["shots"]] == [(0, 2000), (2000, 6000)]
    assert t["meta"]["duration_ms"] == 6000 and t["meta"]["bpm"] == 60.0
    assert t["captions"]["words"][0]["start_ms"] == 200                # shot 1 stretched 1000 -> 2000 ms
    assert t["audio"]["beats_ms"] == []


def test_snap_needs_pattern():
    t = copy.deepcopy(GOOD)
    t["audio"]["beat_pattern"] = []
    with pytest.raises(TemplateError, match="no beat_pattern"):
        snap_template(t, 120.0)


def test_snap_pattern_must_match_shots():
    t = copy.deepcopy(GOOD)
    t["audio"]["beat_pattern"] = [2, 4, 4]
    with pytest.raises(TemplateError, match="beat_pattern has 3 entries but template has 2 shots"):
        snap_template(t, 120.0)
