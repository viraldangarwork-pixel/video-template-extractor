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


@pytest.mark.slow
def test_read_crops_can_downscale(cards_video):
    out = frames.read_crops(cards_video["path"], {0: [("a", (0, 0, 100, 60), 0.5), ("b", (0, 0, 100, 60))]})
    assert out["a"][0].shape == (30, 50, 3) and out["b"][0].shape == (60, 100, 3)
    assert out["a"][0][..., 2].mean() > 200


def test_content_rect_of_letterboxed_and_pillarboxed_sources():
    from vtpl.ingest import content_rect
    assert content_rect(1080, 1920) == {"x": 0, "y": 0, "w": 1080, "h": 1920}
    assert content_rect(540, 960) == {"x": 0, "y": 0, "w": 1080, "h": 1920}
    assert content_rect(1920, 1080) == {"x": 0, "y": 656, "w": 1080, "h": 608}     # 16:9 landscape
    assert content_rect(1920, 1080, rotation=-90) == {"x": 0, "y": 0, "w": 1080, "h": 1920}
    assert content_rect(1080, 1080) == {"x": 0, "y": 420, "w": 1080, "h": 1080}
    assert content_rect(720, 1600) == {"x": 108, "y": 0, "w": 864, "h": 1920}      # narrower than 9:16


def test_cached_content_rect_scales_meta_and_defaults_to_full_frame():
    from vtpl.frames import cached_content
    assert cached_content({"w": 1080, "h": 1920}) == (0, 0, 540, 960)               # meta.json from before `content`
    assert cached_content({"w": 1080, "h": 1920, "content": {"x": 0, "y": 656, "w": 1080, "h": 608}}) == \
        (0, 330, 540, 300)                                                          # inset a few px from the bars
