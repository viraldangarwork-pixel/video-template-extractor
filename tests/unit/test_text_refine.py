import numpy as np

from vtpl.analyzers.text.refine import refine_end, refine_start, text_flags

EMPTY = np.zeros((10, 40, 3), np.uint8)
TEXT = np.full((10, 40, 3), 200, np.uint8)


def ramp(alphas):
    return [(EMPTY * (1 - a) + TEXT * a).astype(np.uint8) for a in alphas]


def test_text_flags_threshold():
    assert text_flags(ramp([0, 0.1, 0.2, 1]), TEXT, EMPTY) == [False, False, True, True]


def test_refine_start_walks_back_to_first_visible_frame():
    crops = ramp([0, 0, 0, 0.3, 0.6, 1, 1])          # frames 10..16, confirmed OCR sample at 15
    ids = list(range(10, 17))
    assert refine_start(ids, crops, 5, TEXT, EMPTY) == 13


def test_refine_end_walks_forward_to_last_visible_frame():
    crops = ramp([1, 1, 1, 0.5, 0.2, 0, 0])          # frames 40..46, confirmed at 41
    ids = list(range(40, 47))
    assert refine_end(ids, crops, 1, TEXT, EMPTY) == 44


def test_confirmed_frame_without_text_is_kept():
    assert refine_start([1, 2, 3], ramp([0, 0, 0]), 2, TEXT, EMPTY) == 3


def test_identical_refs_mean_text_throughout():
    assert refine_start([5, 6, 7], ramp([1, 1, 1]), 2, TEXT, TEXT) == 5
