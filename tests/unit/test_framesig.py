import numpy as np
import pytest

from vtpl.analyzers.framesig import luma, read_small_frames


def test_luma_weights():
    frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
    frames[1, ..., 0] = 1.0  # pure red
    assert luma(frames).tolist() == pytest.approx([0.0, 0.299])


@pytest.mark.slow
def test_read_small_frames(cards_video):
    f = read_small_frames(cards_video["path"])
    assert f.shape[1:] == (64, 36, 3) and f.dtype == np.float32
    assert 178 <= f.shape[0] <= 182          # 6 s at 30 fps
    assert f[0, ..., 0].mean() > 0.8          # first card is red


def test_read_small_frames_bad_file(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError, match="could not decode frames"):
        read_small_frames(bad)
