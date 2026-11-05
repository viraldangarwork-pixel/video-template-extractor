import json

import pytest

from tests.synth.make_fixtures import click_track, solid_clip
from vtpl.render.run import render
from vtpl.util.ffmpeg import duration_ms, probe

pytestmark = pytest.mark.slow


def test_render_produces_video_with_audio(tmp_path):
    from tests.unit.test_schema import GOOD
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(GOOD))
    media = tmp_path / "mine"
    media.mkdir()
    solid_clip(media / "clip_1.mp4", "cyan")
    solid_clip(media / "clip_2.mp4", "magenta")
    click_track(media / "music.wav", seconds=4.0)
    out = tmp_path / "out.mp4"
    warnings = render(tpl, media, out)
    assert warnings == []
    assert abs(duration_ms(out) - 3000) <= 70
    assert {s["codec_type"] for s in probe(out)["streams"]} == {"video", "audio"}
    assert (tmp_path / "render_cmd.txt").read_text().startswith("ffmpeg ")


def _cut_frames(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    labels = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        b, g, r = frame.reshape(-1, 3).mean(axis=0)
        labels.append(g > r)            # cyan has high green, magenta has low green
    cap.release()
    return [k for k in range(1, len(labels)) if labels[k] != labels[k - 1]]


def test_hard_cuts_do_not_drift(tmp_path):
    import copy

    from tests.unit.test_schema import GOOD
    t = copy.deepcopy(GOOD)
    n, length = 6, 1049
    t["shots"] = [{"id": f"s{i + 1}", "start_ms": i * length, "end_ms": (i + 1) * length, "slot": f"clip_{i + 1}",
                   "on_beat": False, "transition_in": {"type": "cut", "duration_ms": 0}} for i in range(n)]
    t["meta"]["duration_ms"] = n * length
    t["audio"]["beat_pattern"] = [2] * n
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(t))
    media = tmp_path / "mine"
    media.mkdir()
    for i in range(n):
        solid_clip(media / f"clip_{i + 1}.mp4", "cyan" if i % 2 == 0 else "magenta")
    out = tmp_path / "out.mp4"
    render(tpl, media, out)
    cuts_ms = [k * 1000 / 30 for k in _cut_frames(out)]
    assert len(cuts_ms) == n - 1
    for i, got in enumerate(cuts_ms, start=1):
        assert abs(got - i * length) <= 34, f"cut {i} at {got:.0f} ms, template says {i * length} ms"


def test_cut_between_landscape_and_portrait_clips(tmp_path):
    import copy

    from tests.synth.make_fixtures import _ff
    from tests.unit.test_schema import GOOD
    t = copy.deepcopy(GOOD)
    t["shots"] = [{"id": f"s{i + 1}", "start_ms": i * 1000, "end_ms": (i + 1) * 1000, "slot": f"clip_{i + 1}",
                   "on_beat": False, "transition_in": {"type": "cut", "duration_ms": 0}} for i in range(2)]
    t["meta"]["duration_ms"] = 2000
    t["audio"]["beat_pattern"] = [2, 2]
    t["captions"]["words"] = []
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(t))
    media = tmp_path / "mine"
    media.mkdir()
    _ff(["-f", "lavfi", "-i", "color=c=cyan:s=1920x1080:r=30:d=3",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(media / "clip_1.mp4")])
    solid_clip(media / "clip_2.mp4", "magenta")
    out = tmp_path / "out.mp4"
    render(tpl, media, out)
    v = next(s for s in probe(out)["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v.get("sample_aspect_ratio", "1:1") == "1:1"
