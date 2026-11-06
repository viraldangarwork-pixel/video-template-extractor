import copy

from tests.unit.test_schema import GOOD, TEXT
from vtpl.fontlib import ass_font_size
from vtpl.render.ass import (ass_color, ass_time, build_ass, caption_lines, escape, overlay_events,
                             replace_words)


def tpl(texts=None, words=None):
    t = copy.deepcopy(GOOD)
    t["texts"] = texts or []
    t["captions"]["words"] = words if words is not None else []
    return t


def item(**anim):
    x = copy.deepcopy(TEXT)
    x["anim"].update(anim)
    return x


def test_time_color_escape():
    assert ass_time(0) == "0:00:00.00" and ass_time(61234) == "0:01:01.23"
    assert ass_color("#FF8000") == "&H000080FF" and ass_color("#FFFFFF", 0x80) == "&H80FFFFFF"
    assert escape("a{b}\\c\nd") == "a(b)/c\\Nd"


def test_style_lines_use_family_and_line_height():
    doc = build_ass(tpl([item()]), {})
    style = next(l for l in doc.splitlines() if l.startswith("Style: t1,"))
    fields = style[len("Style: "):].split(",")
    assert len(fields) == 23 and fields[1] == "Montserrat" and fields[2] == str(ass_font_size("Montserrat-Bold", 80))
    assert fields[7] == "-1" and fields[18] == "5"            # bold, centre alignment


def test_fade_pop_slide_tags():
    ev = overlay_events(item(**{"in": "fade", "in_ms": 200, "out": "fade", "out_ms": 100}), "HI")
    assert len(ev) == 1 and "\\fad(200,100)" in ev[0] and "\\pos(270,1008)" in ev[0]
    ev = overlay_events(item(**{"in": "pop", "in_ms": 150}), "HI")
    assert "\\fscx80\\fscy80\\t(0,100,\\fscx115\\fscy115)\\t(100,150,\\fscx100\\fscy100)" in ev[0]
    ev = overlay_events(item(**{"in": "slide", "in_ms": 200, "in_from": {"dx": -0.25, "dy": 0}}), "HI")
    assert "\\move(0,1008,270,1008,0,200)" in ev[0]


def test_typewriter_one_event_per_char_layout_fixed():
    ev = overlay_events(item(**{"in": "typewriter", "in_ms": 300}), "ABC")
    assert len(ev) == 3
    assert ev[0].split(",", 9)[1:3] == ["0:00:00.10", "0:00:00.20"]
    assert ev[0].endswith("A{\\alpha&HFF&}BC") and ev[2].endswith("ABC")
    assert ev[2].split(",", 9)[2] == "0:00:00.90"


def test_replacement_and_empty_text():
    assert overlay_events(item(), "") == []
    doc = build_ass(tpl([item()]), {"text_1": "NEW"})
    assert doc.rstrip().endswith("NEW")


def test_caption_lines_and_karaoke():
    words = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in
             [("so", 0, 200), ("today", 200, 600), ("we", 700, 800), ("go", 2000, 2300)]]
    lines = caption_lines(words)
    assert [[w["w"] for w in l] for l in lines] == [["so", "today", "we"], ["go"]]
    doc = build_ass(tpl(words=words), {})
    dialogue = [l for l in doc.splitlines() if l.startswith("Dialogue:") and ",Caption," in l]
    assert len(dialogue) == 2
    assert "{\\k20}so {\\k50}today {\\k10}we" in dialogue[0]          # gap after "today" goes to its \k


def test_replace_words_spreads_over_speech_time():
    words = [{"w": "a", "start_ms": 0, "end_ms": 1000}, {"w": "b", "start_ms": 2000, "end_ms": 3000}]
    new = replace_words(words, "xx yy")
    assert new == [{"w": "xx", "start_ms": 0, "end_ms": 1000}, {"w": "yy", "start_ms": 2000, "end_ms": 3000}]
    assert replace_words(words, "") == []


def test_nothing_to_draw_is_none():
    assert build_ass(tpl(), {}) is None


def _scale(event):
    import re
    m = re.search(r"\\fscx(\d+)\\fscy(\d+)", event)
    return (int(m.group(1)), int(m.group(2))) if m else None


def test_long_replacement_is_shrunk_to_fit_the_frame():
    from vtpl.fontlib import pil_font
    from vtpl.render.ass import PLAY_W
    long = "THIS REPLACEMENT IS FAR LONGER THAN THE ORIGINAL"
    ev = overlay_events(item(**{"in": "none", "in_ms": 0}), long)
    sx, sy = _scale(ev[0])
    assert sx == sy < 100                                                       # uniform
    assert pil_font("Montserrat-Bold", 80).getlength(long) * sx / 100 <= 0.9 * PLAY_W
    assert "\\pos(270,1008)" in ev[0]                                           # box centre kept
    assert _scale(overlay_events(item(**{"in": "none", "in_ms": 0}), "SHORT")[0]) is None


def test_shrink_applies_to_pop_and_every_typewriter_step():
    long = "THIS REPLACEMENT IS FAR LONGER THAN THE ORIGINAL"
    s = _scale(overlay_events(item(**{"in": "none", "in_ms": 0}), long)[0])[0]
    pop = overlay_events(item(**{"in": "pop", "in_ms": 150}), long)[0]
    assert f"\\t(100,150,\\fscx{s}\\fscy{s})" in pop and f"\\fscx{round(0.8 * s)}" in pop
    tw = overlay_events(item(**{"in": "typewriter", "in_ms": 300}), long)
    assert all(_scale(e) == (s, s) for e in tw)
