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
