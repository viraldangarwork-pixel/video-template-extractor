from pathlib import Path

from vtpl.analyzers.framesig import luma, read_small_frames
from vtpl.analyzers.transitions import classify_boundaries
from vtpl.util.timebase import frame_to_ms
from vtpl.util.workdir import WorkDir

OUTPUT = "shots.json"
MIN_SHOT_MS = 100


def detect_cuts(video: Path, threshold: float = 3.0, min_scene_len: int = 8) -> list[int]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector

    v = open_video(str(video))
    sm = SceneManager()
    sm.add_detector(AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(v)
    out = []
    for start, _ in sm.get_scene_list()[1:]:
        n = getattr(start, "frame_num", None)          # scenedetect >= 0.7
        out.append(int(n) if n is not None else start.get_frames())
    return out


def build_shots(bounds: list[dict], duration_ms: int, fps: float) -> list[dict]:
    shots, start = [], 0
    tin = {"type": "cut", "duration_ms": 0}
    for b in bounds:
        ms = frame_to_ms(b["frame"], fps)
        if ms - start < MIN_SHOT_MS or duration_ms - ms < MIN_SHOT_MS:
            continue
        shots.append({"start_ms": start, "end_ms": ms, "transition_in": tin})
        start = ms
        tin = {"type": b["type"], "duration_ms": frame_to_ms(b["duration_frames"], fps)}
    shots.append({"start_ms": start, "end_ms": duration_ms, "transition_in": tin})
    return shots


def run(wd: WorkDir, source: Path) -> list[str]:
    meta = wd.read_json("meta.json")
    video = wd.path("norm.mp4")
    frames = read_small_frames(video)
    bounds = classify_boundaries(detect_cuts(video), frames, luma(frames))
    wd.write_json(OUTPUT, {"shots": build_shots(bounds, meta["duration_ms"], meta["fps"])})
    return []
