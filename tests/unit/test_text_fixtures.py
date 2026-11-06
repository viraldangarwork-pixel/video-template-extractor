import numpy as np
import pytest

from tests.synth.text_fixtures import BASIC_ITEMS, TextItem, frame_at, text_crop
from vtpl.util.ffmpeg import duration_ms


def _white_pixels(rgb, cx, cy, half_w=250, half_h=60):
    region = rgb[cy - half_h:cy + half_h, cx - half_w:cx + half_w]
    return int((region.min(axis=2) > 230).sum())


def test_frame_at_shows_items_only_while_visible():
    item = BASIC_ITEMS[0]            # "WAIT FOR IT", 500-3000 ms, white
    assert _white_pixels(frame_at(BASIC_ITEMS, 200), *item.center) == 0
    assert _white_pixels(frame_at(BASIC_ITEMS, 2000), *item.center) > 500


def test_fade_in_is_partial_midway():
    it = TextItem("HELLO", "Montserrat-Bold", 120, (255, 255, 255), (540, 960), 0, 2000, anim_in="fade", anim_ms=400)
    mid, done = frame_at([it], 200), frame_at([it], 600)
    assert _white_pixels(mid, 540, 960) < _white_pixels(done, 540, 960) // 4


def test_typewriter_reveals_left_part_first():
    it = TextItem("SALE", "BebasNeue-Regular", 200, (255, 255, 255), (540, 960), 0, 2000,
                  anim_in="typewriter", anim_ms=400)
    early = frame_at([it], 50)
    cols = np.nonzero((early[860:1060].min(axis=2) > 230).any(axis=0))[0]
    assert len(cols) and cols.max() < 540         # only the first letter, left of centre


def test_text_crop_is_bgr_with_background_border():
    crop = text_crop("HI", "Montserrat-Bold", 80, (255, 255, 255), bg_rgb=(10, 20, 200))
    assert crop.shape[2] == 3 and tuple(crop[0, 0]) == (200, 20, 10)


@pytest.mark.slow
def test_basic_video(text_video_basic):
    assert abs(duration_ms(text_video_basic["path"]) - 4000) <= 50
    assert [t["text"] for t in text_video_basic["truth"]] == ["WAIT FOR IT", "DAY 1", "SALE"]
