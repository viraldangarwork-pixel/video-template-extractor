import pytest
from rapidfuzz import fuzz

from tests.synth.text_fixtures import text_crop
from vtpl import frames, ingest
from vtpl.analyzers.text import detect
from vtpl.util.workdir import WorkDir


def test_ocr_reads_rendered_text():
    img = text_crop("WAIT FOR IT", "Montserrat-Bold", 64, (255, 255, 255), stroke=((0, 0, 0), 4))
    dets = detect.ocr(img)
    best = max(dets, key=lambda d: d["score"])
    assert fuzz.ratio(best["text"].upper(), "WAIT FOR IT") >= 85 and best["score"] >= 0.6
    x, y, w, h = best["box"]
    assert x < 80 and w > img.shape[1] * 0.6 and all(isinstance(v, int) for v in best["box"])


def test_to_norm():
    assert detect.to_norm([54, 96, 270, 48], 540, 960) == [0.1, 0.1, 0.5, 0.05]


@pytest.mark.slow
def test_detect_samples_on_fixture(tmp_path, text_video_basic):
    wd = WorkDir(tmp_path / "w")
    ingest.run(wd, text_video_basic["path"])
    frames.run(wd, text_video_basic["path"])
    samples = detect.detect_samples(wd)
    assert [s["t_ms"] for s in samples[:3]] == [0, 133, 267]
    assert [s["frame"] for s in samples[:3]] == [0, 4, 8]
    late = next(s for s in samples if s["t_ms"] >= 2400)
    texts = [d["text"].upper() for d in late["dets"]]
    for want in ["WAIT FOR IT", "DAY 1", "SALE"]:
        assert any(fuzz.ratio(t, want) >= 80 for t in texts), (want, texts)
    for d in late["dets"]:
        assert all(0 <= v <= 1 for v in d["box"]) and d["score"] >= detect.MIN_SCORE
