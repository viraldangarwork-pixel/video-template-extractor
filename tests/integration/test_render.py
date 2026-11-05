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
