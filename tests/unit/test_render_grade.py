import pytest

from vtpl.render.grade import eq_filter, eq_params

CLIP = {"luma_mean": 0.4, "luma_std": 0.2, "chroma": 0.05}
PIVOT = 112 / 219        # eq's contrast pivot (luma code 128) in full-range units


def test_eq_params_maps_clip_to_target():
    p = eq_params({"luma_mean": 0.55, "luma_std": 0.25, "chroma": 0.08}, CLIP)
    assert p["contrast"] == pytest.approx(1.25) and p["saturation"] == pytest.approx(1.6)
    assert p["brightness"] == pytest.approx((0.55 - ((0.4 - PIVOT) * 1.25 + PIVOT)) * 219 / 255, abs=5e-4)


def test_eq_brightness_is_in_limited_range_codes():
    # a full-range mean shift of 219/255 * x needs eq brightness x (eq adds codes on the 16..235 scale)
    p = eq_params({"luma_mean": 0.4 + 0.2 * 255 / 219, "luma_std": 0.2, "chroma": 0.05}, CLIP)
    assert p == {"brightness": 0.2, "contrast": 1.0, "saturation": 1.0}
    # contrast alone pivots on code 128, so a clip centred there keeps its mean
    centred = {"luma_mean": PIVOT, "luma_std": 0.2, "chroma": 0.05}
    assert eq_params({"luma_mean": PIVOT, "luma_std": 0.26, "chroma": 0.05}, centred)["brightness"] == 0.0


def test_eq_params_clamped_and_neutral():
    p = eq_params({"luma_mean": 1.0, "luma_std": 0.9, "chroma": 0.5}, CLIP)
    assert p == {"brightness": 0.3, "contrast": 1.5, "saturation": 1.8}
    assert eq_params(CLIP, CLIP) is None
    flat = {"luma_mean": 0.4, "luma_std": 0.0, "chroma": 0.0}       # nothing to scale: contrast stays 1
    assert eq_params({"luma_mean": 0.6, "luma_std": 0.2, "chroma": 0.05}, flat)["contrast"] == 1.0


def test_eq_filter():
    assert eq_filter({"brightness": 0.1, "contrast": 1.2, "saturation": 0.9}) == \
        "eq=brightness=0.100:contrast=1.200:saturation=0.900"
