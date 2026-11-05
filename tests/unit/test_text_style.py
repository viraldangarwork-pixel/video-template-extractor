import numpy as np
import pytest

from tests.synth.text_fixtures import text_crop
from vtpl.analyzers.text.style import analyze_style, combine_styles, delta_e, hex_from_bgr, ink_height, to_lab


def lab_of_hex(h):
    rgb = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return to_lab(np.array([rgb[::-1]], np.uint8))[0]


def close(hex_a, hex_b, tol=15):
    return delta_e(lab_of_hex(hex_a), lab_of_hex(hex_b)) <= tol


def test_white_fill_black_stroke():
    s = analyze_style(text_crop("WAIT FOR IT", "Montserrat-Bold", 96, (255, 255, 255), stroke=((0, 0, 0), 6)))
    assert close(hex_from_bgr(s.color_bgr), "#FFFFFF")
    assert s.stroke_bgr is not None and close(hex_from_bgr(s.stroke_bgr), "#000000") and 4 <= s.stroke_px <= 8
    assert not s.shadow


def test_yellow_with_shadow_no_stroke():
    s = analyze_style(text_crop("DAY 1", "Anton-Regular", 110, (255, 214, 0), shadow=True))
    assert close(hex_from_bgr(s.color_bgr), "#FFD600") and s.stroke_bgr is None and s.shadow


def test_red_fill_white_stroke():
    s = analyze_style(text_crop("SALE", "BebasNeue-Regular", 160, (230, 30, 60), stroke=((255, 255, 255), 4)))
    assert close(hex_from_bgr(s.color_bgr), "#E61E3C")
    assert s.stroke_bgr is not None and close(hex_from_bgr(s.stroke_bgr), "#FFFFFF")


def test_plain_text_has_no_stroke_from_antialiasing():
    s = analyze_style(text_crop("hello", "Poppins-Regular", 90, (255, 255, 255)))
    assert s.stroke_bgr is None and not s.shadow


def test_ink_height_and_combine():
    s = analyze_style(text_crop("SALE", "BebasNeue-Regular", 160, (230, 30, 60)))
    assert 100 <= ink_height(s.fill_mask) <= 130            # Bebas cap height is ~0.7 em
    combined = combine_styles([s, s, s])
    assert combined["stroke"] is None and combined["shadow"] is False and combined["color"].startswith("#")


def test_solid_crop_raises():
    with pytest.raises(ValueError):
        analyze_style(np.full((60, 200, 3), 90, np.uint8))
