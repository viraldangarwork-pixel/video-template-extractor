import pytest

from vtpl.pipeline import STAGE_NAMES, Stage, analyze


def _stages(calls, fail=None, with_shots=False):
    def make(name, output):
        def fn(wd, source):
            calls.append(name)
            if name == fail:
                raise RuntimeError(f"{name} broke")
            wd.write_json(output, {"ok": True})
            return [f"{name} warn"] if name == "audio" else []
        return Stage(name, output, fn)
    shots = [make("shots", "shots.json")] if with_shots else []
    return [make("ingest", "meta.json"), *shots, make("audio", "audio.json"), make("merge", "template.json")]


def test_default_stage_order():
    assert STAGE_NAMES == ["ingest", "frames", "shots", "audio", "speech", "text", "merge"]


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
    assert calls[3:] == ["audio", "merge"]                             # merge re-runs after audio


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


def test_merge_reruns_after_failed_stage_recovers(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    calls = []
    analyze(src, tmp_path / "work", stages=_stages(calls, fail="audio"))
    assert calls == ["ingest", "audio", "merge"]
    analyze(src, tmp_path / "work", stages=_stages(calls))
    assert calls == ["ingest", "audio", "merge", "audio", "merge"]
    analyze(src, tmp_path / "work", stages=_stages(calls))
    assert len(calls) == 5                                             # fully cached now


def test_only_stage_triggers_merge(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    calls = []
    analyze(src, tmp_path / "work", stages=_stages(calls))
    del calls[:]
    analyze(src, tmp_path / "work", force=True, only={"audio"}, stages=_stages(calls))
    assert calls == ["audio", "merge"]


def test_forced_merge_skipped_until_its_inputs_exist(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    calls = []
    out = analyze(src, tmp_path / "work", only={"ingest"}, stages=_stages(calls, with_shots=True))
    assert calls == ["ingest"] and not out.exists()                    # no shots.json yet: merge skipped
    analyze(src, tmp_path / "work", only={"shots"}, stages=_stages(calls, with_shots=True))
    assert calls == ["ingest", "shots", "merge"] and out.exists()
