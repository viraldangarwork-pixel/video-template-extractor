import json

import pytest

from tests.synth.make_fixtures import texture_clip, texture_image, zoom_clip
from vtpl.pipeline import analyze

pytestmark = pytest.mark.slow
STAGES = {"ingest", "frames", "shots", "motion", "grade", "merge"}


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    p = tmp_path_factory.mktemp("tex") / "tex.png"
    texture_image(p)
    return p


def _template(tmp_path, video):
    import json
    return json.loads(analyze(video, tmp_path / "work", only=STAGES).read_text())


def test_zoom_detected(tmp_path, image):
    video = tmp_path / "zoom.mp4"
    zoom_clip(video, image, 3.0, 1.25)
    t = _template(tmp_path, video)
    assert len(t["shots"]) == 1
    m = t["shots"][0]["motion"]
    assert m["type"] == "zoom_in" and m["scale_to"] == pytest.approx(1.25, abs=0.06)


def test_still_is_static_and_graded(tmp_path, image):
    video = tmp_path / "still.mp4"
    texture_clip(video, image, 3.0)
    t = _template(tmp_path, video)
    assert t["shots"][0]["motion"]["type"] == "static"
    g = t["shots"][0]["grade"]
    assert 0.2 < g["luma_mean"] < 0.8 and g["luma_std"] > 0.05


def test_eq_brightening_shows_in_grade(tmp_path, image):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    texture_clip(a, image, 2.0)
    texture_clip(b, image, 2.0, "eq=brightness=0.15:saturation=1.6")
    ga = _template(tmp_path / "A", a)["shots"][0]["grade"]
    gb = _template(tmp_path / "B", b)["shots"][0]["grade"]
    assert gb["luma_mean"] > ga["luma_mean"] + 0.08 and gb["chroma"] > ga["chroma"] * 1.3


def test_letterboxed_grade_measures_only_the_picture(tmp_path, image):
    import cv2

    from vtpl.analyzers.grade import stats
    video = tmp_path / "landscape.mp4"
    texture_clip(video, image, 2.0, "transpose=1")             # 1920x1080: ingest pads it with black bars
    cap = cv2.VideoCapture(str(video))
    frames = []
    while len(frames) < 10:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    unpadded = stats(frames)
    t = _template(tmp_path, video)
    meta = json.loads((next((tmp_path / "work").iterdir()) / "meta.json").read_text())
    assert meta["content"]["w"] == 1080 and 600 <= meta["content"]["h"] <= 610
    g = t["shots"][0]["grade"]
    for k in ("luma_mean", "luma_std", "chroma"):
        assert g[k] == pytest.approx(unpadded[k], abs=0.03), (k, g, unpadded)
    assert t["shots"][0]["motion"]["type"] == "static"
