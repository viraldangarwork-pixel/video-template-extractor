import json

import numpy as np
import pytest
from rapidfuzz import fuzz

from tests.synth.make_fixtures import solid_clip
from vtpl.analyzers.text.style import delta_e, to_lab
from vtpl.pipeline import analyze
from vtpl.render.run import render

pytestmark = pytest.mark.slow
STAGES = {"ingest", "frames", "shots", "text", "merge"}


def lab(h):
    return to_lab(np.array([[int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)]], np.uint8))[0]


def overlays(t):
    return sorted((x for x in t["texts"] if x["kind"] == "overlay"), key=lambda x: x["start_ms"])


def test_text_roundtrip(tmp_path, text_video_basic):
    tpl_path = analyze(text_video_basic["path"], tmp_path / "work", only=STAGES)
    tpl = json.loads(tpl_path.read_text())
    media = tmp_path / "mine"
    media.mkdir()
    for shot in tpl["shots"]:
        solid_clip(media / f"{shot['slot']}.mp4", "darkslategray", seconds=5)
    out = tmp_path / "out.mp4"
    render(tpl_path, media, out)
    again = json.loads(analyze(out, tmp_path / "work2", only=STAGES).read_text())

    a, b = overlays(tpl), overlays(again)
    assert len(a) == len(b) == 3
    for x, y in zip(a, b):
        assert fuzz.ratio(x["sample"].upper(), y["sample"].upper()) >= 80
        assert abs(x["start_ms"] - y["start_ms"]) <= 100 and abs(x["end_ms"] - y["end_ms"]) <= 100
        for k in ("x", "y"):
            assert abs((x["box"][k] + x["box"]["w" if k == "x" else "h"] / 2)
                       - (y["box"][k] + y["box"]["w" if k == "x" else "h"] / 2)) <= 0.02
        assert delta_e(lab(x["style"]["color"]), lab(y["style"]["color"])) <= 15
        assert (x["style"]["stroke"] is None) == (y["style"]["stroke"] is None)
        assert x["anim"]["in"] == y["anim"]["in"]
        assert x["style"]["font"] == y["style"]["font"] or x["style"]["font"] == y["style"].get("font_guess")
