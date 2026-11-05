import numpy as np

from vtpl.analyzers.framesig import luma
from vtpl.analyzers.transitions import (
    classify_boundaries, dissolve_span, find_fades, find_flashes, runs,
)

RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
BLUE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
GREEN = np.array([0.0, 0.5, 0.0], dtype=np.float32)


def solid(color, n):
    return np.broadcast_to(color, (n, 8, 4, 3)).astype(np.float32).copy()


def blend(a, b, n):
    alphas = np.linspace(1, 0, n + 2)[1:-1]
    return np.stack([np.broadcast_to(al * a + (1 - al) * b, (8, 4, 3)) for al in alphas]).astype(np.float32)


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]


def test_hard_cut_stays_cut():
    frames = np.concatenate([solid(RED, 30), solid(BLUE, 30)])
    b = classify_boundaries([30], frames, luma(frames))
    assert [(x["frame"], x["type"], x["duration_frames"]) for x in b] == [(30, "cut", 0)]


def test_dissolve_detected_from_detector_cut():
    frames = np.concatenate([solid(RED, 30), blend(RED, BLUE, 15), solid(BLUE, 30)])
    span = dissolve_span(frames, 37)
    assert span is not None and abs(span[0] - 30) <= 1 and abs(span[1] - 44) <= 1
    b = classify_boundaries([37], frames, luma(frames))
    assert len(b) == 1 and b[0]["type"] == "dissolve" and abs(b[0]["frame"] - 37) <= 1
    assert 13 <= b[0]["duration_frames"] <= 15


def test_dissolve_found_without_detector_cut():
    frames = np.concatenate([solid(RED, 30), blend(RED, BLUE, 15), solid(BLUE, 30)])
    b = classify_boundaries([], frames, luma(frames))
    assert [x["type"] for x in b] == ["dissolve"]


def test_fade_through_black():
    black = np.zeros(3, dtype=np.float32)
    frames = np.concatenate([solid(BLUE, 30), blend(BLUE, black, 7), solid(black, 2),
                             blend(black, GREEN, 7), solid(GREEN, 30)])
    fades = find_fades(luma(frames))
    assert len(fades) == 1 and abs(fades[0]["frame"] - 38) <= 1 and fades[0]["type"] == "fade"
    b = classify_boundaries([31, 45], frames, luma(frames))   # detector noise inside the fade
    assert [x["type"] for x in b] == ["fade"]


def test_video_starting_black_is_not_a_fade():
    black = np.zeros(3, dtype=np.float32)
    frames = np.concatenate([solid(black, 5), blend(black, BLUE, 6), solid(BLUE, 30)])
    assert find_fades(luma(frames)) == []


def test_flash_between_shots():
    white = np.ones(3, dtype=np.float32)
    frames = np.concatenate([solid(GREEN, 30), blend(GREEN, white, 2), solid(white, 2),
                             blend(white, BLUE, 2), solid(BLUE, 30)])
    fl = find_flashes(luma(frames), frames)
    assert len(fl) == 1 and fl[0]["type"] == "flash" and abs(fl[0]["frame"] - 33) <= 1


def test_flash_within_same_shot_ignored():
    white = np.ones(3, dtype=np.float32)
    frames = np.concatenate([solid(GREEN, 30), solid(white, 2), solid(GREEN, 30)])
    assert find_flashes(luma(frames), frames) == []


def test_long_dissolve_bracketed_by_detector_cuts_is_one_dissolve():
    frames = np.concatenate([solid(RED, 30), blend(RED, BLUE, 25), solid(BLUE, 30)])
    b = classify_boundaries([30, 54], frames, luma(frames))
    assert [x["type"] for x in b] == ["dissolve"]
    assert 22 <= b[0]["duration_frames"] <= 25


def test_very_long_dissolve_without_detector_cuts_is_one_dissolve():
    frames = np.concatenate([solid(RED, 30), blend(RED, BLUE, 40), solid(BLUE, 30)])
    b = classify_boundaries([], frames, luma(frames))
    assert [x["type"] for x in b] == ["dissolve"]
