import cv2
import numpy as np
import pytest

from tests.synth.text_fixtures import text_crop
from vtpl.analyzers.text.font import crop_to_ink, em_size_for, ink_mask, rank_fonts, similarity
from vtpl.analyzers.text.style import analyze_style, ink_height


def video_like(crop):
    """Blur + JPEG round-trip, like a frame decoded from a compressed video."""
    blurred = cv2.GaussianBlur(crop, (3, 3), 0.8)
    ok, buf = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_crop_to_ink_and_ink_mask_height():
    m = np.zeros((20, 30), bool)
    m[5:9, 3:7] = True
    assert crop_to_ink(m).shape == (4, 4)
    assert ink_mask("SALE", "BebasNeue-Regular", 57).shape[0] == 57


def test_similarity_identity_beats_other_font():
    a = ink_mask("WAIT FOR IT", "Montserrat-Bold", 60)
    assert similarity(a, a) == pytest.approx(1.0)
    assert similarity(a, ink_mask("WAIT FOR IT", "PlayfairDisplay-Bold", 60)) < 0.8


@pytest.mark.parametrize("text,font_id,size", [
    ("WAIT FOR IT", "Montserrat-Bold", 96),
    ("SALE TODAY", "BebasNeue-Regular", 140),
    ("Golden Hour", "PlayfairDisplay-Bold", 100),
    ("day one", "Pacifico-Regular", 100),
])
def test_true_font_in_top3(text, font_id, size):
    s = analyze_style(video_like(text_crop(text, font_id, size, (255, 255, 255), stroke=((0, 0, 0), 4))))
    ranked = rank_fonts(text, s.fill_mask)
    assert len(ranked) == 3 and font_id in [r[0] for r in ranked], ranked


def test_em_size_recovered():
    s = analyze_style(text_crop("WAIT FOR IT", "Montserrat-Bold", 96, (255, 255, 255)))
    assert abs(em_size_for("WAIT FOR IT", "Montserrat-Bold", ink_height(s.fill_mask)) - 96) <= 4


def test_empty_mask_raises():
    with pytest.raises(ValueError, match="empty text mask"):
        rank_fonts("X", np.zeros((10, 10), bool))


def test_ocr_spacing_errors_do_not_decide_the_font():
    # OCR often merges or splits words ("LINKIN BIO" for "LINK IN BIO"); the glyphs still match.
    s = analyze_style(video_like(text_crop("LINK IN BIO", "Montserrat-Bold", 80, (255, 255, 0))))
    assert rank_fonts("LINKIN BIO", s.fill_mask)[0][0] == "Montserrat-Bold"
    s = analyze_style(video_like(text_crop("DAY 1", "Anton-Regular", 110, (255, 214, 0))))
    assert rank_fonts("DAY1", s.fill_mask)[0][0] == "Anton-Regular"
