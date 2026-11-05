import json

import pytest

from tests.synth.make_fixtures import solid_clip
from vtpl.pipeline import analyze
from vtpl.render.run import render

pytestmark = pytest.mark.slow
NO_SPEECH = {"ingest", "shots", "audio", "merge"}
COLORS = ["cyan", "magenta", "orange", "teal", "olive"]
FRAME_MS = 34


def _roundtrip(tmp_path, video, with_music=None):
    tpl_path = analyze(video, tmp_path / "work", only=NO_SPEECH)
    tpl = json.loads(tpl_path.read_text())
    media = tmp_path / "mine"
    media.mkdir()
    for i, shot in enumerate(tpl["shots"]):
        solid_clip(media / f"{shot['slot']}.mp4", COLORS[i % len(COLORS)])
    if with_music:
        (media / "music.wav").write_bytes(with_music.read_bytes())
    out = tmp_path / "out.mp4"
    render(tpl_path, media, out)
    again = json.loads(analyze(out, tmp_path / "work2", only=NO_SPEECH).read_text())
    return tpl, again


def test_roundtrip_hard_cuts(tmp_path, cards_video):
    tpl, again = _roundtrip(tmp_path, cards_video["path"], with_music=cards_video["wav"])
    assert [s["on_beat"] for s in tpl["shots"]] == [False, True, True, True]
    assert len(again["shots"]) == len(tpl["shots"])
    for a, b in zip(tpl["shots"], again["shots"]):
        assert abs(a["start_ms"] - b["start_ms"]) <= FRAME_MS
    assert abs(again["meta"]["bpm"] - tpl["meta"]["bpm"]) <= 2


def test_roundtrip_transitions(tmp_path, transitions_video):
    tpl, again = _roundtrip(tmp_path, transitions_video["path"])
    assert [s["transition_in"]["type"] for s in again["shots"]] == [s["transition_in"]["type"] for s in tpl["shots"]]
    for a, b in zip(tpl["shots"], again["shots"]):
        assert abs(a["start_ms"] - b["start_ms"]) <= 2 * FRAME_MS
