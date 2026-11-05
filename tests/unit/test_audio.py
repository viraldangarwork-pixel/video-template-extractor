import pytest

from vtpl.analyzers.audio import analyze_beats, beat_pattern, is_on_beat
from tests.synth.make_fixtures import click_signal


def test_click_track_beats_and_bpm():
    y, sr, truth = click_signal(seconds=8.0, bpm=120.0)
    res = analyze_beats(y, sr)
    assert abs(res["bpm"] - 120.0) <= 2.0
    matched = sum(any(abs(b - t) <= 30 for b in res["beats_ms"]) for t in truth)
    assert matched >= 0.8 * len(truth)
    assert len(res["downbeat_hint_ms"]) >= len(res["beats_ms"]) // 4


def test_silence_has_no_beats():
    import numpy as np
    res = analyze_beats(np.zeros(22050 * 3, dtype=np.float32), 22050)
    assert res == {"bpm": None, "beats_ms": [], "downbeat_hint_ms": []}


def test_is_on_beat():
    beats = [500, 1000, 1500]
    assert is_on_beat(1070, beats)
    assert not is_on_beat(1250, beats)
    assert not is_on_beat(1000, [])


def test_beat_pattern():
    shots = [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 1000, "end_ms": 1200},
             {"start_ms": 1200, "end_ms": 3200}]
    assert beat_pattern(shots, period_ms=500) == [2, 1, 4]


@pytest.mark.slow
def test_beats_of_file_reads_aac_m4a(tmp_path, monkeypatch):
    import tempfile

    from tests.synth.make_fixtures import _ff, click_track
    from vtpl.analyzers.audio import beats_of_file
    wav, m4a = tmp_path / "click.wav", tmp_path / "music.m4a"
    click_track(wav, seconds=8.0, bpm=120.0)
    _ff(["-i", str(wav), "-c:a", "aac", "-b:a", "128k", str(m4a)])
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    res = beats_of_file(m4a)
    assert res["bpm"] is not None and abs(res["bpm"] - 120.0) <= 2.0
    assert list(scratch.iterdir()) == []            # temporary WAV cleaned up
