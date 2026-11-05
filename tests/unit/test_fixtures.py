import pytest

from tests.synth.make_fixtures import CARDS, click_signal
from vtpl.util.ffmpeg import duration_ms, probe


def test_click_signal_beats():
    y, sr, beats = click_signal(seconds=4.0, bpm=120.0)
    assert sr == 22050 and len(y) == 4 * 22050
    assert beats == [500, 1000, 1500, 2000, 2500, 3000, 3500]


@pytest.mark.slow
def test_cards_video(cards_video):
    assert cards_video["cuts_ms"] == [1500, 2500, 4000]
    assert abs(duration_ms(cards_video["path"]) - int(sum(d for _, d in CARDS) * 1000)) <= 50
    kinds = {s["codec_type"] for s in probe(cards_video["path"])["streams"]}
    assert kinds == {"video", "audio"}


@pytest.mark.slow
def test_transitions_video(transitions_video):
    assert [t["type"] for t in transitions_video["truth"]] == ["dissolve", "fade", "flash"]
    assert [t["ms"] for t in transitions_video["truth"]] == [1750, 3250, 4900]
