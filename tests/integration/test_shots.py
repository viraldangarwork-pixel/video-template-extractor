import pytest

from vtpl import ingest
from vtpl.analyzers import shots
from vtpl.util.workdir import WorkDir

pytestmark = pytest.mark.slow
FRAME_MS = 34


def _analyse(tmp_path, video):
    wd = WorkDir(tmp_path / "w")
    ingest.run(wd, video)
    shots.run(wd, video)
    return wd.read_json("shots.json")["shots"]


def test_hard_cuts(tmp_path, cards_video):
    got = [s["start_ms"] for s in _analyse(tmp_path, cards_video["path"])[1:]]
    assert len(got) == len(cards_video["cuts_ms"])
    for g, t in zip(got, cards_video["cuts_ms"]):
        assert abs(g - t) <= FRAME_MS


def test_transitions(tmp_path, transitions_video):
    got = _analyse(tmp_path, transitions_video["path"])[1:]
    truth = transitions_video["truth"]
    assert [s["transition_in"]["type"] for s in got] == [t["type"] for t in truth]
    for s, t in zip(got, truth):
        assert abs(s["start_ms"] - t["ms"]) <= 2 * FRAME_MS
        assert abs(s["transition_in"]["duration_ms"] - t["duration_ms"]) <= 100
