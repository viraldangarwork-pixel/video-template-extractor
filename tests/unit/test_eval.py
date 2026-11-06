import pytest

from vtpl.eval import format_report, match_times, score, summarise


def test_match_times():
    assert match_times([1000, 2000, 3000], [1010, 2100, 3020], 34) == (2, 1, 1)
    assert match_times([], [], 34) == (0, 0, 0)


TEMPLATE = {"meta": {"bpm": 119.5},
            "shots": [{"start_ms": 0, "motion": {"type": "static"}}, {"start_ms": 1500, "motion": {"type": "zoom_in"}}],
            "texts": [{"kind": "overlay", "sample": "WAIT FOR IT", "start_ms": 520, "end_ms": 2990},
                      {"kind": "caption", "sample": "hello", "start_ms": 0, "end_ms": 100}]}


def test_score_each_field():
    s = score({"cuts_ms": [1500, 2500], "bpm": 120, "texts": [{"text": "WAIT FOR IT", "start_ms": 500, "end_ms": 3000}],
               "motion": [{"shot": 2, "type": "zoom_in"}, {"shot": 1, "type": "pan"}]}, TEMPLATE)
    assert s["cuts"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3)}
    assert s["bpm"] == {"expected": 120, "got": 119.5, "ok": True}
    assert s["texts"] == {"found": 1.0, "timing_ok": 1.0}
    assert s["motion"] == {"accuracy": 0.5}


def test_score_skips_missing_fields_and_summary():
    assert score({}, TEMPLATE) == {}
    report = {"a": {"cuts": {"f1": 1.0}, "bpm": {"ok": True}}, "b": {"cuts": {"f1": 0.5}, "bpm": {"ok": False}}}
    summary = summarise(report)
    assert summary == {"cuts_f1": 0.75, "bpm_ok": 0.5}
    assert "cuts_f1" in format_report(report, summary)


def _golden(tmp_path):
    import json
    g = tmp_path / "golden"
    g.mkdir(parents=True)
    (g / "a.mp4").write_bytes(b"")
    (g / "a.expected.json").write_text(json.dumps({"cuts_ms": [1500]}))
    return g


def _fake_analyze(seen, tmp_path):
    import json

    def fake(video, out, force=False, only=None):
        seen.append(force)
        tpl = tmp_path / "tpl.json"
        tpl.write_text(json.dumps(TEMPLATE))
        return tpl
    return fake


def test_evaluate_reanalyses_by_default(tmp_path, monkeypatch):
    from vtpl import eval as ev
    seen = []
    monkeypatch.setattr(ev, "analyze", _fake_analyze(seen, tmp_path))
    ev.evaluate(_golden(tmp_path), tmp_path / "work")
    ev.evaluate(_golden(tmp_path / "x"), tmp_path / "work", force=False)
    assert seen == [True, False]


def test_eval_cli_force_flag(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from vtpl import cli
    from vtpl import eval as ev
    seen = []
    monkeypatch.setattr(ev, "analyze", _fake_analyze(seen, tmp_path))
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    g = _golden(tmp_path)
    for flags in ([], ["--no-force"], ["--force"]):
        r = CliRunner().invoke(cli.app, ["eval", "--golden", str(g), "--out", str(tmp_path / "w"), *flags])
        assert r.exit_code == 0, r.output
    assert seen == [True, False, True]
