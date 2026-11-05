import shutil

import pytest

from tests.synth import make_fixtures as mf


def _need_ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def cards_video(fixtures_dir):
    _need_ffmpeg()
    seconds = sum(d for _, d in mf.CARDS)
    wav = fixtures_dir / "clicks.wav"
    beats = mf.click_track(wav, seconds=seconds)
    path = fixtures_dir / "cards.mp4"
    cuts = mf.color_cards(path, mf.CARDS, audio=wav)
    return {"path": path, "cuts_ms": cuts, "beats_ms": beats, "wav": wav}


@pytest.fixture(scope="session")
def small_cards_video(fixtures_dir):
    _need_ffmpeg()
    path = fixtures_dir / "small_cards.mp4"
    wav = fixtures_dir / "clicks_small.wav"
    mf.click_track(wav, seconds=6.0)
    mf.color_cards(path, mf.CARDS, audio=wav, size="540x960")
    return path


@pytest.fixture(scope="session")
def transitions_video(fixtures_dir):
    _need_ffmpeg()
    path = fixtures_dir / "transitions.mp4"
    return {"path": path, "truth": mf.transitions(path)}


@pytest.fixture(scope="session")
def text_video_basic(fixtures_dir):
    _need_ffmpeg()
    from tests.synth.text_fixtures import BASIC_ITEMS, text_video
    path = fixtures_dir / "text_basic.mp4"
    return {"path": path, "truth": text_video(path, BASIC_ITEMS, 4.0)}
