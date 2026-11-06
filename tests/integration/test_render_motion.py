import copy
import json

import pytest

from tests.synth.make_fixtures import texture_clip, texture_image
from tests.unit.test_schema import GOOD
from vtpl.pipeline import analyze
from vtpl.render.run import render

pytestmark = pytest.mark.slow
STAGES = {"ingest", "frames", "shots", "motion", "merge"}


def _one_shot(motion):
    t = copy.deepcopy(GOOD)
    t["shots"] = [{"id": "s1", "start_ms": 0, "end_ms": 3000, "slot": "clip_1", "on_beat": False,
                   "transition_in": {"type": "cut", "duration_ms": 0}, "motion": motion}]
    t["meta"]["duration_ms"] = 3000
    t["audio"]["beat_pattern"] = []
    t["captions"]["words"] = []          # no burned-in captions over the measured picture
    return t


@pytest.mark.parametrize("motion,check", [
    ({"type": "zoom_in", "scale_from": 1.0, "scale_to": 1.25, "dx": 0.0, "dy": 0.0},
     lambda m: m["type"] == "zoom_in" and abs(m["scale_to"] - 1.25) <= 0.06),
    ({"type": "pan", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.1, "dy": 0.0},
     lambda m: m["type"] == "pan" and abs(m["dx"] - 0.1) <= 0.03 and abs(m["dy"]) <= 0.02),
])
def test_rendered_motion_is_measured_back(tmp_path, motion, check):
    image = tmp_path / "tex.png"
    texture_image(image)
    media = tmp_path / "mine"
    media.mkdir()
    texture_clip(media / "clip_1.mp4", image, 4.0)
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps(_one_shot(motion)))
    out = tmp_path / "out.mp4"
    render(tpl, media, out)
    again = json.loads(analyze(out, tmp_path / "work", only=STAGES).read_text())
    assert check(again["shots"][0]["motion"]), again["shots"][0]["motion"]


def _ramp_clip(path, fps, seconds):
    """A clip whose (uniform) brightness rises linearly with time, so a frame's level tells its source time."""
    import subprocess

    import numpy as np
    n = round(fps * seconds)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "270x480",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                          "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(np.full((480, 270), round(30 + 50 * i / fps), np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0


def _frame_levels(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(float(f.mean()))
    cap.release()
    return out


def test_motion_on_60fps_clip_plays_at_real_speed(tmp_path):
    media = tmp_path / "mine"
    media.mkdir()
    _ramp_clip(media / "clip_1.mp4", 60, 4.0)        # level 30 at 0 s .. 230 at 4 s
    levels = {}
    for name, motion in (("zoom", {"type": "zoom_in", "scale_from": 1.0, "scale_to": 1.2, "dx": 0.0, "dy": 0.0}),
                         ("still", {"type": "static", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.0, "dy": 0.0})):
        t = _one_shot(motion)
        t["shots"][0]["end_ms"] = t["meta"]["duration_ms"] = 2000
        tpl = tmp_path / f"{name}.json"
        tpl.write_text(json.dumps(t))
        render(tpl, media, tmp_path / f"{name}.mp4")
        levels[name] = _frame_levels(tmp_path / f"{name}.mp4")
    zoom, still = levels["zoom"], levels["still"]
    assert len(zoom) == len(still) == 60
    # the 2 s shot is cut from the clip's middle (1 s..3 s): its brightness rises by ~100 levels at real speed
    assert still[-1] - still[0] == pytest.approx(50 * 59 / 30, abs=8)
    assert zoom[-1] - zoom[0] == pytest.approx(still[-1] - still[0], abs=6), (zoom[0], zoom[-1])
    assert max(abs(a - b) for a, b in zip(zoom, still)) < 6
