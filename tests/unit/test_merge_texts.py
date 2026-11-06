import copy

import pytest

from vtpl import merge
from vtpl.schema import TemplateError, validate

META = {"w": 1080, "h": 1920, "fps": 30, "duration_ms": 4000, "has_audio": True}
SHOTS = {"shots": [{"start_ms": 0, "end_ms": 4000, "transition_in": {"type": "cut", "duration_ms": 0}}]}


def track(tid, text, start, end, conf=0.8):
    return {"id": tid, "text": text, "start_ms": start, "end_ms": end,
            "box": {"x": 0.1, "y": 0.6, "w": 0.8, "h": 0.05},
            "style": {"color": "#FFFFFF", "stroke": {"color": "#000000", "px": 5}, "shadow": False, "size_px": 90},
            "font": {"id": "BebasNeue-Regular", "confidence": conf, "candidates": [["BebasNeue-Regular", conf]]},
            "anim": {"in": "pop", "in_ms": 150, "out": "fade", "out_ms": 200}}


WORDS = [{"w": "so", "start_ms": 1000, "end_ms": 1200, "p": 0.9}, {"w": "today", "start_ms": 1200, "end_ms": 1600, "p": 0.9},
         {"w": "we", "start_ms": 1700, "end_ms": 1800, "p": 0.9}]


def test_is_caption():
    assert merge.is_caption(track("t1", "SO TODAY", 1000, 1600), WORDS)
    assert not merge.is_caption(track("t1", "WAIT FOR IT", 1000, 1600), WORDS)
    assert not merge.is_caption(track("t1", "SO TODAY", 3000, 3500), WORDS)     # wrong time


def test_texts_overlays_slots_and_caption_ref():
    tracks = [track("t1", "SO TODAY", 1000, 1600), track("t2", "SALE", 500, 2500), track("t3", "DAY 1", 2000, 3000)]
    t = merge.build_template(META, SHOTS, None, {"words": WORDS}, "abc", {"tracks": tracks})
    validate(t)
    overlays = [x for x in t["texts"] if x["kind"] == "overlay"]
    assert [(x["id"], x["slot"], x["sample"]) for x in overlays] == [("t1", "text_1", "SALE"), ("t2", "text_2", "DAY 1")]
    cap = next(x for x in t["texts"] if x["kind"] == "caption")
    assert cap["id"] == "c1" and t["captions"]["style_ref"] == "c1" and cap["sample"] == "SO TODAY"
    assert overlays[0]["style"] == {"font": "BebasNeue-Regular", "font_confidence": 0.8, "size_px": 90,
                                    "color": "#FFFFFF", "stroke": {"color": "#000000", "px": 5}, "shadow": False}


def test_low_confidence_font_falls_back():
    t = merge.build_template(META, SHOTS, None, None, "abc", {"tracks": [track("t1", "SALE", 0, 1000, conf=0.3)]})
    st = t["texts"][0]["style"]
    assert st["font"] == "Montserrat-Bold" and st["font_guess"] == "BebasNeue-Regular"


def test_texts_clipped_to_duration_and_dropped_if_empty():
    tracks = [track("t1", "A LONG ONE", 3000, 4500), track("t2", "LATE", 4100, 4600)]
    t = merge.build_template(META, SHOTS, None, None, "abc", {"tracks": tracks})
    assert [(x["sample"], x["end_ms"]) for x in t["texts"]] == [("A LONG ONE", 4000)]


def test_run_reads_text_json(tmp_path):
    from vtpl.util.workdir import WorkDir
    wd = WorkDir(tmp_path / "abc")
    wd.write_json("meta.json", META)
    wd.write_json("shots.json", SHOTS)
    wd.write_json("text.json", {"tracks": [track("t1", "SALE", 0, 1000)]})
    merge.run(wd, tmp_path / "v.mp4")
    assert wd.read_json("template.json")["texts"][0]["sample"] == "SALE"


def test_colours_are_normalised_to_uppercase():
    tr = track("t1", "SALE", 0, 1000)
    tr["style"].update(color="#ffcc00", stroke={"color": "#00ff7f", "px": 5})
    st = merge.build_template(META, SHOTS, None, None, "abc", {"tracks": [tr]})["texts"][0]["style"]
    assert st["color"] == "#FFCC00" and st["stroke"]["color"] == "#00FF7F"
