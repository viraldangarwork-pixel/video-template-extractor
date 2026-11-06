import cv2
import numpy as np
import pytest

from vtpl.analyzers.motion import STATIC, classify, shot_window, track_motion

H, W = 960, 540


def texture(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (H, W), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    for _ in range(60):
        x, y = rng.integers(0, W - 40), rng.integers(0, H - 40)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 30, int(y) + 30), int(rng.integers(0, 255)), -1)
    return img


def warp(img, scale=1.0, tx=0.0, ty=0.0):
    m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REFLECT)


def test_zoom_in_sequence():
    base = texture()
    frames = [warp(base, 1.02 ** i) for i in range(10)]
    scale, dx, dy = track_motion(frames)
    assert scale == pytest.approx(1.02 ** 9, abs=0.02) and abs(dx) < 0.01 and abs(dy) < 0.01


def test_pan_sequence():
    base = texture()
    frames = [warp(base, tx=4 * i) for i in range(10)]
    scale, dx, dy = track_motion(frames)
    assert scale == pytest.approx(1.0, abs=0.01) and dx == pytest.approx(36 / W, abs=0.01) and abs(dy) < 0.01


def test_flat_frames_have_no_motion_estimate():
    assert track_motion([np.zeros((H, W), np.uint8)] * 5) is None


def test_classify():
    assert classify(1.1, 0, 0)["type"] == "zoom_in"
    assert classify(0.9, 0, 0)["type"] == "zoom_out"
    assert classify(1.0, 0.05, 0) == {"type": "pan", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.05, "dy": 0.0}
    assert classify(1.01, 0.01, 0.0) == STATIC


def test_shot_window_skips_transitions_and_edges():
    shots = [{"start_ms": 0, "end_ms": 2000, "transition_in": {"type": "cut", "duration_ms": 0}},
             {"start_ms": 2000, "end_ms": 4000, "transition_in": {"type": "dissolve", "duration_ms": 400}}]
    assert shot_window(shots, 0, 15) == (1, 25)        # ends 200 ms before the dissolve's centre, minus 1 frame
    assert shot_window(shots, 1, 15) == (34, 58)
