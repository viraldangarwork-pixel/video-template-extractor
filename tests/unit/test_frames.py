import pytest

from vtpl import frames, ingest
from vtpl.util.workdir import WorkDir


def test_clamp_rect():
    assert frames.clamp_rect((-10, -10, 50, 50), 100, 100) == (0, 0, 40, 40)
    assert frames.clamp_rect((90, 95, 50, 50), 100, 100) == (90, 95, 10, 5)
    assert frames.clamp_rect((200, 0, 10, 10), 100, 100) == (100, 0, 0, 10)


def test_cache_time_ms():
    assert frames.cache_time_ms(30) == 2000 and frames.cache_time_ms(1) == 67


@pytest.mark.slow
def test_frames_stage(tmp_path, small_cards_video):
    wd = WorkDir(tmp_path / "w")
    ingest.run(wd, small_cards_video)
    assert frames.run(wd, small_cards_video) == []
    doc = wd.read_json("frames.json")
    assert doc["fps"] == 15 and (doc["w"], doc["h"]) == (540, 960)
    assert 88 <= len(doc["files"]) <= 92                      # 6 s at 15 fps
    first = frames.cached_frame(wd, 0)
    assert first.shape == (960, 540, 3) and first[..., 2].mean() > 200     # red card, BGR


@pytest.mark.slow
def test_read_crops_single_pass(cards_video):
    req = {0: [("a", (0, 0, 100, 100))],
           50: [("a", (10, 10, 50, 50)), ("b", (-20, -20, 60, 60))],
           130: [("c", (1000, 1800, 500, 500))]}
    out = frames.read_crops(cards_video["path"], req)
    assert out["a"][0].shape == (100, 100, 3) and out["a"][0][..., 2].mean() > 200   # red
    assert out["a"][50][..., 0].mean() > 200                                          # blue
    assert out["b"][50].shape == (40, 40, 3)
    assert out["c"][130].shape == (120, 80, 3)                                        # clamped at frame edge
    assert frames.read_crops(cards_video["path"], {}) == {}
