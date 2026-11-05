import pytest

from tests.synth.text_fixtures import TextItem, frame_at
from vtpl.analyzers.text.anim import classify, measure

STEP = 1000 / 30
BAND = slice(860, 1060)


def band(rgb):
    return rgb[BAND, :, ::-1].copy()


def seq_in(item, n=10):
    crops = [band(frame_at([item], round(item.start_ms + k * STEP))) for k in range(n)]
    return crops, band(frame_at([], 0))


def seq_out(item, n=10):
    crops = [band(frame_at([item], round(item.end_ms - 1 - k * STEP))) for k in range(n)]
    return crops, band(frame_at([], 0))


def item(**kw):
    base = dict(text="HELLO", font_id="Montserrat-Bold", size=110, color=(255, 255, 255), center=(540, 960),
                start_ms=0, end_ms=3000, anim_ms=200)
    return TextItem(**{**base, **kw})


@pytest.mark.parametrize("anim", ["pop", "fade", "slide", "typewriter", "none"])
def test_in_animations(anim):
    it = item(anim_in=anim, anim_ms=300 if anim == "typewriter" else 200)
    crops, empty = seq_in(it)
    res = classify(measure(crops, (255, 255, 255), empty), 1080, 1920, STEP)
    assert res["kind"] == anim
    if anim == "slide":
        assert res["from"]["dx"] == pytest.approx(-300 / 1080, abs=0.05) and abs(res["from"]["dy"]) < 0.01
    if anim != "none":
        assert 100 <= res["ms"] <= 400


@pytest.mark.parametrize("anim", ["pop", "fade", "none"])
def test_out_animations(anim):
    crops, empty = seq_out(item(anim_out=anim))
    assert classify(measure(crops, (255, 255, 255), empty), 1080, 1920, STEP)["kind"] == anim


def test_no_text_is_none():
    crops = [band(frame_at([], 0))] * 5
    assert classify(measure(crops, (255, 255, 255), crops[0]), 1080, 1920, STEP) == {"kind": "none", "ms": 0}
