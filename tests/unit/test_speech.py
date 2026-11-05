import os
from types import SimpleNamespace as NS

import pytest

from vtpl.analyzers import speech
from vtpl.util.workdir import WorkDir


def test_words_from_segments():
    segs = [NS(words=[NS(word=" Hello", start=0.3, end=0.52, probability=0.91),
                      NS(word=" ", start=0.52, end=0.53, probability=0.1)]),
            NS(words=None),
            NS(words=[NS(word="world!", start=0.6, end=1.0, probability=0.8)])]
    assert speech.words_from_segments(segs) == [
        {"w": "Hello", "start_ms": 300, "end_ms": 520, "p": 0.91},
        {"w": "world!", "start_ms": 600, "end_ms": 1000, "p": 0.8},
    ]


def test_no_audio_skips(tmp_path):
    wd = WorkDir(tmp_path)
    wd.write_json("meta.json", {"has_audio": False})
    assert speech.run(wd, tmp_path / "x.mp4") == ["no audio: speech skipped"]
    assert wd.read_json("speech.json") == {"language": None, "words": []}


@pytest.mark.skipif(os.environ.get("VTPL_RUN_WHISPER") != "1", reason="set VTPL_RUN_WHISPER=1 (downloads a model)")
def test_whisper_runs_on_clicks(tmp_path):
    from tests.synth.make_fixtures import click_track
    wd = WorkDir(tmp_path)
    click_track(wd.path("audio.wav"), seconds=3.0)
    wd.write_json("meta.json", {"has_audio": True})
    os.environ.setdefault("VTPL_WHISPER_MODEL", "tiny")
    assert speech.run(wd, tmp_path / "x.mp4") == []
    assert isinstance(wd.read_json("speech.json")["words"], list)
