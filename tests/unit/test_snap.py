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


def _three_shots(pattern):
    t = copy.deepcopy(GOOD)
    t["shots"] = [{"id": f"s{i + 1}", "start_ms": i * 1000, "end_ms": (i + 1) * 1000, "slot": f"clip_{i + 1}",
                   "on_beat": False, "transition_in": {"type": "cut", "duration_ms": 0}} for i in range(3)]
    t["meta"]["duration_ms"] = 3000
    t["audio"]["beat_pattern"] = pattern
    return t


def test_snap_aligns_cuts_to_beat_phase():
    t = snap_template(_three_shots([2, 2, 1]), new_bpm=120.0, first_beat_ms=250)
    validate(t)
    cuts = [s["end_ms"] for s in t["shots"]]                   # both cuts and the end of the last shot
    assert cuts == [1250, 2250, 2750]
    assert all(c % 500 == 250 for c in cuts)
    assert t["shots"][0]["start_ms"] == 0 and t["meta"]["duration_ms"] == 2750


def test_snap_phase_uses_first_beat_modulo_period():
    a = snap_template(_three_shots([2, 2, 1]), new_bpm=120.0, first_beat_ms=1250)
    b = snap_template(_three_shots([2, 2, 1]), new_bpm=120.0, first_beat_ms=250)
    assert a["shots"] == b["shots"]


def test_snap_grid_does_not_drift_with_fractional_period():
    t = _three_shots([1, 1, 1])
    t = snap_template(t, new_bpm=90.0, first_beat_ms=100)      # period 666.67 ms
    assert [s["end_ms"] for s in t["shots"]] == [round(100 + k * 60000 / 90) for k in (1, 2, 3)]
