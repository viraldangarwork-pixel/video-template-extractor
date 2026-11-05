import pytest

from vtpl.pipeline import STAGE_NAMES, Stage, analyze


def _stages(calls, fail=None):
    def make(name, output):
        def fn(wd, source):
            calls.append(name)
            if name == fail:
                raise RuntimeError(f"{name} broke")
            wd.write_json(output, {"ok": True})
            return [f"{name} warn"] if name == "audio" else []
        return Stage(name, output, fn)
    return [make("ingest", "meta.json"), make("audio", "audio.json"), make("merge", "template.json")]


def test_default_stage_order():
    assert STAGE_NAMES == ["ingest", "shots", "audio", "speech", "merge"]


def test_runs_all_then_caches(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    calls = []
    out = analyze(src, tmp_path / "work", stages=_stages(calls))
    assert calls == ["ingest", "audio", "merge"] and out.name == "template.json"
    run = (out.parent / "run.json").read_text()
    assert "audio warn" in run
    analyze(src, tmp_path / "work", stages=_stages(calls))
    assert calls == ["ingest", "audio", "merge"]                       # all cached
    analyze(src, tmp_path / "work", force=True, only={"audio"}, stages=_stages(calls))
    assert calls[-1] == "audio" and len(calls) == 4


def test_non_critical_failure_is_recorded(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    out = analyze(src, tmp_path / "work", stages=_stages([], fail="audio"))
    import json
    run = json.loads((out.parent / "run.json").read_text())
    assert run["stages"]["audio"] == {"status": "failed", "seconds": pytest.approx(0, abs=1), "warnings": ["audio broke"]}
    assert run["stages"]["merge"]["status"] == "ok"


def test_critical_failure_raises(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    with pytest.raises(RuntimeError, match="ingest broke"):
        analyze(src, tmp_path / "work", stages=_stages([], fail="ingest"))
