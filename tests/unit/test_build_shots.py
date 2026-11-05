from vtpl.analyzers.shots import build_shots


def test_build_shots_from_boundaries():
    bounds = [{"frame": 45, "type": "cut", "duration_frames": 0},
              {"frame": 90, "type": "dissolve", "duration_frames": 15},
              {"frame": 91, "type": "cut", "duration_frames": 0}]       # 33 ms shot: dropped
    shots = build_shots(bounds, duration_ms=5000, fps=30)
    assert shots == [
        {"start_ms": 0, "end_ms": 1500, "transition_in": {"type": "cut", "duration_ms": 0}},
        {"start_ms": 1500, "end_ms": 3000, "transition_in": {"type": "cut", "duration_ms": 0}},
        {"start_ms": 3000, "end_ms": 5000, "transition_in": {"type": "dissolve", "duration_ms": 500}},
    ]


def test_build_shots_no_boundaries():
    assert build_shots([], duration_ms=2000, fps=30) == [
        {"start_ms": 0, "end_ms": 2000, "transition_in": {"type": "cut", "duration_ms": 0}}]
