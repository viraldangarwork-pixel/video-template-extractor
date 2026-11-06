import numpy as np
import pytest
from rapidfuzz import fuzz

from vtpl import frames, ingest
from vtpl.analyzers.text import stage
from vtpl.analyzers.text.style import delta_e, to_lab
from vtpl.util.workdir import WorkDir

pytestmark = pytest.mark.slow


def lab(h):
    return to_lab(np.array([[int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)]], np.uint8))[0]


def _run_stage(tmp_path_factory, video):
    wd = WorkDir(tmp_path_factory.mktemp("text") / "w")
    ingest.run(wd, video["path"])
    frames.run(wd, video["path"])
    assert stage.run(wd, video["path"]) == []
    return wd.read_json("text.json")["tracks"]


@pytest.fixture(scope="module")
def tracks(tmp_path_factory, text_video_basic):
    return _run_stage(tmp_path_factory, text_video_basic)


@pytest.fixture(scope="module")
def moving_tracks(tmp_path_factory, text_video_moving):
    return _run_stage(tmp_path_factory, text_video_moving)


def test_one_track_per_text(tracks, text_video_basic):
    truth = text_video_basic["truth"]
    assert len(tracks) == len(truth)
    for t in truth:
        assert any(fuzz.ratio(tr["text"].upper(), t["text"]) >= 80 for tr in tracks), t["text"]


def _match(tracks, t):
    return max(tracks, key=lambda tr: fuzz.ratio(tr["text"].upper(), t["text"]))


def test_timing_position_colour_stroke_shadow(tracks, text_video_basic):
    for t in text_video_basic["truth"]:
        tr = _match(tracks, t)
        assert abs(tr["start_ms"] - t["start_ms"]) <= 100, (t["text"], tr["start_ms"])
        assert abs(tr["end_ms"] - t["end_ms"]) <= 100, (t["text"], tr["end_ms"])
        b = tr["box"]
        assert abs(b["x"] + b["w"] / 2 - t["center"][0]) <= 0.02
        assert abs(b["y"] + b["h"] / 2 - t["center"][1]) <= 0.02
        assert delta_e(lab(tr["style"]["color"]), lab(t["color"])) <= 15
        assert (tr["style"]["stroke"] is not None) == t["stroke"]
        assert tr["style"]["shadow"] == t["shadow"]


def test_font_size_and_animation(tracks, text_video_basic):
    for t in text_video_basic["truth"]:
        tr = _match(tracks, t)
        assert t["font"] in [c[0] for c in tr["font"]["candidates"]], (t["text"], tr["font"])
        if tr["font"]["id"] == t["font"]:
            assert abs(tr["style"]["size_px"] - t["size"]) <= 0.08 * t["size"]
        assert tr["anim"]["in"] == t["anim_in"], (t["text"], tr["anim"])
        assert tr["anim"]["out"] == t["anim_out"], (t["text"], tr["anim"])


def test_moving_background_timing_and_animation(moving_tracks, text_video_moving):
    """Background motion must not shift timings or turn hard cuts into fades; a text visible from the
    first frame or to the last has no in / out animation."""
    truth = text_video_moving["truth"]
    assert len(moving_tracks) == len(truth)
    for t in truth:
        tr = _match(moving_tracks, t)
        assert abs(tr["start_ms"] - t["start_ms"]) <= 100, (t["text"], tr["start_ms"])
        assert abs(tr["end_ms"] - t["end_ms"]) <= 100, (t["text"], tr["end_ms"])
        assert (tr["anim"]["in"], tr["anim"]["out"]) == (t["anim_in"], t["anim_out"]), (t["text"], tr["anim"])
