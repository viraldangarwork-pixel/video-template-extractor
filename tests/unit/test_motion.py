import math

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


def test_extrapolation_is_capped_and_motion_bounded():
    from vtpl.analyzers.motion import MAX_DXY, MAX_K, MAX_SCALE, extrapolate
    assert MAX_K == 3 and MAX_SCALE == 1.5 and MAX_DXY == 0.25
    scale, dx, dy = extrapolate(1.05, 0.02, -0.01, 10.0)          # a short window can't be stretched 10x
    assert scale == pytest.approx(1.05 ** 3) and dx == pytest.approx(0.06) and dy == pytest.approx(-0.03)
    scale, dx, dy = extrapolate(0.496, 0.4, -3.30, 1.0)            # the testsrc2 case: wild estimates are bounded
    assert scale == pytest.approx(1 / 1.5) and dx == 0.25 and dy == -0.25
    assert extrapolate(2.0, 0.0, 0.0, 1.2)[0] == 1.5
    assert extrapolate(1.1, 0.01, 0.0, 1.5) == pytest.approx((1.1 ** 1.5, 0.015, 0.0))


def _shake_offsets(n, amp, seed=1):
    """Zero-drift handheld tremor: a sum of sines (px) sampled at the 15 fps cache rate."""
    import math
    ph = np.random.default_rng(seed).uniform(0, 6.28, 4)
    out = []
    for i in range(n):
        t = i / 15
        out.append((amp * (math.sin(2 * math.pi * 1.3 * t + ph[0]) + 0.5 * math.sin(2 * math.pi * 4.1 * t + ph[1])),
                    amp * (math.sin(2 * math.pi * 0.9 * t + ph[2]) + 0.5 * math.sin(2 * math.pi * 3.7 * t + ph[3]))))
    return out


def test_handheld_shake_is_not_a_pan():
    from vtpl.analyzers.motion import window_motion
    base = texture()
    offsets = _shake_offsets(31, 0.023 * W)
    frames = [warp(base, tx=ox, ty=oy) for ox, oy in offsets]
    assert classify(*track_motion(frames))["type"] == "pan"        # summed displacement looks like a pan
    assert classify(*window_motion(frames)) == STATIC


def test_steady_pan_is_still_a_pan():
    from vtpl.analyzers.motion import window_motion
    base = texture()
    frames = [warp(base, tx=4 * i) for i in range(10)]
    scale, dx, dy = window_motion(frames)
    assert scale == pytest.approx(1.0, abs=0.01) and dx == pytest.approx(36 / W, abs=0.01) and abs(dy) < 0.01
    assert classify(scale, dx, dy)["type"] == "pan"
    shaky = [warp(base, tx=4 * i + ox / 3, ty=oy / 3) for i, (ox, oy) in enumerate(_shake_offsets(30, 0.023 * W))]
    m = classify(*window_motion(shaky))
    assert m["type"] == "pan" and m["dx"] == pytest.approx(116 / W, abs=0.02)


def test_steady_zoom_and_line_fit():
    from vtpl.analyzers.motion import steady_motion, window_motion
    base = texture()
    scale, dx, dy = window_motion([warp(base, 1.02 ** i) for i in range(10)])
    assert scale == pytest.approx(1.02 ** 9, abs=0.02) and abs(dx) < 0.01 and abs(dy) < 0.01
    # per-pair (index, dlog scale, dx, dy): a back-and-forth wobble has no steady trend
    wobble = np.array([[i, 0.01 * (-1) ** i, 0.004 * (-1) ** i, 0.0] for i in range(20)])
    assert steady_motion(wobble, 20) == (1.0, 0.0, 0.0)
    steady = np.array([[i, math.log(1.01), 0.002, -0.001] for i in range(20)])
    assert steady_motion(steady, 20) == pytest.approx((1.01 ** 20, 0.04, -0.02))


def test_untracked_pairs_are_extrapolated_not_counted_as_still():
    from vtpl.analyzers.motion import steady_motion
    rows = [[i, math.log(1.01), 0.002, 0.0] for i in list(range(5)) + list(range(10, 20))]   # pairs 5-9 lost
    assert steady_motion(np.array(rows), 20) == pytest.approx((1.01 ** 20, 0.04, 0.0))
    base = texture()
    frames = [warp(base, tx=4 * i) for i in range(10)]
    frames[5] = np.zeros_like(base)                # two pairs can't be tracked
    scale, dx, dy = track_motion(frames)
    assert dx == pytest.approx(36 / W, abs=0.01)
