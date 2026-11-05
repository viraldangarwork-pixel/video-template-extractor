import pytest

from vtpl import ingest
from vtpl.util.ffmpeg import probe
from vtpl.util.workdir import WorkDir

pytestmark = pytest.mark.slow


def test_ingest_normalises_and_extracts_audio(tmp_path, small_cards_video):
    wd = WorkDir(tmp_path / "w")
    warnings = ingest.run(wd, small_cards_video)
    assert warnings == []
    v = probe(wd.path("norm.mp4"))["streams"][0]
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v["r_frame_rate"] == "30/1"
    assert wd.has("audio.wav")
    meta = wd.read_json("meta.json")
    assert meta["has_audio"] is True and meta["fps"] == 30
    assert abs(meta["duration_ms"] - 6000) <= 50


def test_ingest_without_audio(tmp_path, transitions_video):
    wd = WorkDir(tmp_path / "w")
    assert ingest.run(wd, transitions_video["path"]) == ["no audio stream"]
    assert wd.read_json("meta.json")["has_audio"] is False
    assert not wd.has("audio.wav")


def test_ingest_rejects_non_video(tmp_path):
    wav = tmp_path / "a.wav"
    from tests.synth.make_fixtures import click_track
    click_track(wav, seconds=1.0)
    with pytest.raises(ValueError, match="has no video stream"):
        ingest.run(WorkDir(tmp_path / "w"), wav)
