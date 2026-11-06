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
    (lambda t: (t["shots"][0].update(end_ms=10), t["shots"][1].update(start_ms=10),
                t["shots"][1]["transition_in"].update(type="cut", duration_ms=0)),
     "shots\\[0\\] covers no whole frame at 30 fps"),
    (lambda t: (t["shots"][1].update(end_ms=1050), t["shots"][1]["transition_in"].update(type="cut", duration_ms=0),
                t["shots"].append(
        {"id": "s3", "start_ms": 1050, "end_ms": 1083, "slot": "clip_3", "on_beat": False,    # 50-83 ms: frames 2..2
         "transition_in": {"type": "cut", "duration_ms": 0}})), "shots\\[2\\] covers no whole frame at 30 fps"),
])
def test_timeline_errors(mutate, msg):
    t = copy.deepcopy(GOOD)
    mutate(t)
    with pytest.raises(TemplateError, match=msg):
        validate(t)


def test_one_frame_shot_is_allowed():
    t = copy.deepcopy(GOOD)
    t["shots"][0]["end_ms"] = t["shots"][1]["start_ms"] = 33           # frame-rounded one frame at 30 fps
    t["shots"][1]["transition_in"] = {"type": "cut", "duration_ms": 0}
    validate(t)


def test_clamp_transitions():
    shots = copy.deepcopy(GOOD["shots"])
    shots[1]["transition_in"]["duration_ms"] = 900
    clamp_transitions(shots)
    assert shots[1]["transition_in"] == {"type": "dissolve", "duration_ms": 500}
    shots[1]["transition_in"] = {"type": "fade", "duration_ms": 0}
    clamp_transitions(shots)
    assert shots[1]["transition_in"] == {"type": "cut", "duration_ms": 0}


TEXT = {"id": "t1", "slot": "text_1", "sample": "HI", "kind": "overlay", "start_ms": 100, "end_ms": 900,
        "box": {"x": 0.1, "y": 0.5, "w": 0.3, "h": 0.05},
        "style": {"font": "Montserrat-Bold", "font_confidence": 0.9, "size_px": 80, "color": "#FFFFFF",
                  "stroke": None, "shadow": False},
        "anim": {"in": "fade", "in_ms": 200, "out": "none", "out_ms": 0}}


def test_text_item_valid_and_errors():
    t = copy.deepcopy(GOOD)
    t["texts"] = [copy.deepcopy(TEXT)]
    validate(t)
    t["texts"][0]["anim"]["in"] = "spin"
    with pytest.raises(TemplateError, match="texts/0/anim/in"):
        validate(t)
    t["texts"][0]["anim"]["in"] = "fade"
    t["texts"][0]["end_ms"] = 5000
    with pytest.raises(TemplateError, match=r"texts\[0\] must lie within"):
        validate(t)


def test_unknown_font_is_a_template_error():
    t = copy.deepcopy(GOOD)
    t["texts"] = [copy.deepcopy(TEXT)]
    t["texts"][0]["style"]["font"] = "Comic-Sans"
    with pytest.raises(TemplateError, match=r"texts/0/style/font: unknown font 'Comic-Sans'; see .*catalog\.json"):
        validate(t)


def test_lowercase_hex_colours_are_accepted():
    t = copy.deepcopy(GOOD)
    t["texts"] = [copy.deepcopy(TEXT)]
    t["texts"][0]["style"].update(color="#ffcc00", stroke={"color": "#00ff7f", "px": 3})
    validate(t)
    t["texts"][0]["style"]["color"] = "#ffcc0"
    with pytest.raises(TemplateError, match="texts/0/style/color"):
        validate(t)


def test_dangling_caption_style_ref_is_rejected():
    t = copy.deepcopy(GOOD)
    t["captions"]["style_ref"] = "c1"
    with pytest.raises(TemplateError, match="captions/style_ref"):
        validate(t)
    cap = copy.deepcopy(TEXT)
    cap.update(id="c1", slot="captions", kind="caption")
    t["texts"] = [cap]
    validate(t)
