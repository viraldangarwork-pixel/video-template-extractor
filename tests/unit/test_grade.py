import numpy as np
import pytest

from vtpl.analyzers.grade import frame_stats, sample_indices, stats


def test_gray_frame():
    y, sd, c = frame_stats(np.full((20, 20, 3), 128, np.uint8))
    assert y == pytest.approx(128 / 255, abs=0.01) and sd == pytest.approx(0, abs=1e-6) and c < 0.01


def test_contrast_and_chroma_increase():
    flat = np.full((20, 20, 3), 128, np.uint8)
    stripes = flat.copy()
    stripes[::2] = 30
    stripes[1::2] = 230
    red = np.zeros((20, 20, 3), np.uint8)
    red[..., 2] = 220
    assert frame_stats(stripes)[1] > 0.3 > frame_stats(flat)[1]
    assert frame_stats(red)[2] > 0.15


def test_stats_is_median_over_frames():
    frames = [np.full((10, 10, 3), v, np.uint8) for v in (50, 128, 200)]
    s = stats(frames)
    assert s["luma_mean"] == pytest.approx(128 / 255, abs=0.01) and set(s) == {"luma_mean", "luma_std", "chroma"}


def test_sample_indices():
    assert sample_indices(10, 30, 5) == [10, 15, 20, 25, 30]
    assert sample_indices(4, 5, 5) == [4, 5]
