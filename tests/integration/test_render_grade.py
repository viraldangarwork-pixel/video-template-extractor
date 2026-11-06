import copy
import json

import pytest

from tests.synth.make_fixtures import texture_clip, texture_image
from tests.unit.test_schema import GOOD
from vtpl.pipeline import analyze
from vtpl.render.run import render

pytestmark = pytest.mark.slow
STAGES = {"ingest", "frames", "shots", "grade", "merge"}


def test_grade_round_trip(tmp_path):
    image = tmp_path / "tex.png"
    texture_image(image)
    graded = tmp_path / "graded.mp4"
    texture_clip(graded, image, 3.0, "eq=brightness=0.12:contrast=1.3:saturation=1.6")
    target = json.loads(analyze(graded, tmp_path / "w1", only=STAGES).read_text())["shots"][0]["grade"]

    t = copy.deepcopy(GOOD)
    t["shots"] = [{"id": "s1", "start_ms": 0, "end_ms": 3000, "slot": "clip_1", "on_beat": False,
                   "transition_in": {"type": "cut", "duration_ms": 0}, "grade": target}]
    t["meta"]["duration_ms"] = 3000
    t["audio"]["beat_pattern"] = []
    t["captions"]["words"] = []
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps(t))
    media = tmp_path / "mine"
    media.mkdir()
    texture_clip(media / "clip_1.mp4", image, 4.0)             # the same picture, ungraded
    out = tmp_path / "out.mp4"
    render(tpl, media, out)
    got = json.loads(analyze(out, tmp_path / "w2", only=STAGES).read_text())["shots"][0]["grade"]
    for k in ("luma_mean", "luma_std", "chroma"):
        assert got[k] == pytest.approx(target[k], abs=0.03), (k, got, target)
