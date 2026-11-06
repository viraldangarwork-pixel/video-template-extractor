import copy
import json

import pytest
from rapidfuzz import fuzz

from tests.synth.make_fixtures import solid_clip
from tests.unit.test_schema import GOOD, TEXT
from vtpl import frames
from vtpl.analyzers.text.detect import ocr
from vtpl.render.run import render

pytestmark = pytest.mark.slow


def _setup(tmp_path, texts_yaml=""):
    media = tmp_path / "mine"
    media.mkdir()
    solid_clip(media / "clip_1.mp4", "navy")
    solid_clip(media / "clip_2.mp4", "maroon")
    if texts_yaml:
        (media / "texts.yaml").write_text(texts_yaml)
    return media


def _read(out, frame_index):
    return frames.read_crops(out, {frame_index: [("f", (0, 0, 1080, 1920))]})["f"][frame_index]


def _texts(img):
    return [d["text"].upper() for d in ocr(img)]


def test_overlay_text_is_burned_with_replacement(tmp_path):
    t = copy.deepcopy(GOOD)
    item = copy.deepcopy(TEXT)
    item.update(start_ms=500, end_ms=2500, sample="HELLO", box={"x": 0.2, "y": 0.45, "w": 0.6, "h": 0.06})
    item["style"].update(size_px=140, stroke={"color": "#000000", "px": 6})
    t["texts"] = [item]
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(t))
    out = tmp_path / "out dir:1" / "out.mp4"
    render(tpl, _setup(tmp_path, "text_1: WORLD\n"), out)
    assert out.with_suffix(".ass").exists()
    assert any(fuzz.ratio(x, "WORLD") >= 80 for x in _texts(_read(out, 45)))       # 1.5 s
    assert not any(fuzz.ratio(x, "WORLD") >= 80 for x in _texts(_read(out, 6)))    # 0.2 s, before start


def test_captions_are_burned(tmp_path):
    t = copy.deepcopy(GOOD)
    t["captions"]["words"] = [{"w": "hello", "start_ms": 300, "end_ms": 900},
                              {"w": "there", "start_ms": 900, "end_ms": 1500}]
    cap = copy.deepcopy(TEXT)                       # the source showed captions in this style
    cap.update(id="c1", slot="captions", kind="caption", sample="HELLO THERE", start_ms=300, end_ms=1500,
               box={"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.06})
    cap["style"].update(size_px=110, stroke={"color": "#000000", "px": 5})
    t["texts"] = [cap]
    t["captions"]["style_ref"] = "c1"
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(t))
    out = tmp_path / "out.mp4"
    render(tpl, _setup(tmp_path), out)
    assert any("HELLO" in x for x in _texts(_read(out, 36)))                       # 1.2 s


def test_speech_without_source_captions_is_not_burned(tmp_path):
    t = copy.deepcopy(GOOD)
    t["captions"]["words"] = [{"w": "hello", "start_ms": 300, "end_ms": 900}]
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(t))
    out = tmp_path / "out.mp4"
    render(tpl, _setup(tmp_path), out)
    assert not out.with_suffix(".ass").exists()
