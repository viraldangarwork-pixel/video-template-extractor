import cv2
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


def _yuv420(bgr):
    """Round trip through 4:2:0 chroma subsampling, as encoded video has it."""
    return cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420), cv2.COLOR_YUV2BGR_I420)


@pytest.mark.parametrize("phase_ms", [0, 5, 10, 15, 20, 25, 30])
def test_mixed_case_fade_in_is_not_a_pop(phase_ms):
    # During a fade only some glyph pixels reach the fill colour first (here: subsampled chroma and
    # the slightly off-white fill measured from video), so the fill mask is shorter than the text;
    # that is not a pop, which scales width and height together.
    it = item(text="Line two here", font_id="Poppins-Bold", size=70, anim_in="fade", anim_ms=300)
    crops = [_yuv420(band(frame_at([it], round(it.start_ms + phase_ms + k * STEP)))) for k in range(10)]
    res = classify(measure(crops, (255, 252, 252), _yuv420(band(frame_at([], 0)))), 1080, 1920, STEP)
    assert res["kind"] == "fade" and 100 <= res["ms"] <= 400, res


def test_fade_with_a_short_partial_mask_is_not_a_pop():
    # Measurements seen on the encoded fixture: no fill pixels while faint, then one frame whose
    # fill mask has the full width but 84 % of the height.
    ms = [{"n": 0, "w": 0.0, "h": 0.0, "cx": 0.0, "cy": 0.0, "contrast": c} for c in (0.12, 0.23, 0.34, 0.45, 0.57, 0.67)]
    ms += [{"n": n, "w": 470.0, "h": h, "cx": 540.0, "cy": 40.0, "contrast": c}
           for n, h, c in ((6112, 63.0, 0.78), (9553, 75.0, 0.90), (10281, 75.0, 1.0), (10283, 75.0, 1.0))]
    assert classify(ms, 1080, 1920, STEP)["kind"] == "fade"
