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


@pytest.mark.parametrize("content,msg", [
    ("{bad", "a.expected.json: not valid JSON"),
    ("[1, 2]", "a.expected.json: must be a JSON object"),
    ('{"texts": [{"text": "HI"}]}', "a.expected.json: texts\\[0\\] needs"),
    ('{"motion": [{"type": "pan"}]}', "a.expected.json: motion\\[0\\] needs"),
    ('{"cuts_ms": [1000, "x"]}', "a.expected.json: cuts_ms"),
    ('{"bpm": "fast"}', "a.expected.json: bpm"),
])
def test_malformed_expected_files_fail_before_analysis(tmp_path, monkeypatch, content, msg):
    from vtpl import eval as ev
    seen = []
    monkeypatch.setattr(ev, "analyze", _fake_analyze(seen, tmp_path))
    g = _golden(tmp_path)
    (g / "b.mp4").write_bytes(b"")
    (g / "b.expected.json").write_text("{}")
    (g / "a.expected.json").write_text(content)
    with pytest.raises(ValueError, match=msg):
        ev.evaluate(g, tmp_path / "work")
    assert seen == []                                  # nothing analysed before every file checked out


def test_missing_video_is_skipped_with_a_warning(tmp_path, monkeypatch):
    import json

    from vtpl import eval as ev
    monkeypatch.setattr(ev, "analyze", _fake_analyze([], tmp_path))
    g = _golden(tmp_path)
    (g / "x.expected.json").write_text("{}")
    result = ev.evaluate(g, tmp_path / "work")
    assert list(result["videos"]) == ["a"]
    assert any("x.expected.json" in w and "skipped" in w for w in result["warnings"])
    assert json.loads((tmp_path / "work" / "eval_report.json").read_text())["warnings"] == result["warnings"]


def test_report_lists_failed_stages(tmp_path, monkeypatch):
    import json

    from vtpl import eval as ev
    monkeypatch.setattr(ev, "analyze", _fake_analyze([], tmp_path))
    (tmp_path / "run.json").write_text(json.dumps({"stages": {
        "shots": {"status": "ok", "warnings": []},
        "text": {"status": "failed", "warnings": ["boom: no OCR model"]}}}))
    result = ev.evaluate(_golden(tmp_path), tmp_path / "work")
    assert result["videos"]["a"]["failed_stages"] == {"text": "boom: no OCR model"}
    assert "failed_stages" in format_report(result["videos"], result["summary"])


def test_eval_cli_prints_warnings(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from vtpl import cli
    from vtpl import eval as ev
    monkeypatch.setattr(ev, "analyze", _fake_analyze([], tmp_path))
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    g = _golden(tmp_path)
    (g / "x.expected.json").write_text("{}")
    r = CliRunner().invoke(cli.app, ["eval", "--golden", str(g), "--out", str(tmp_path / "w")])
    assert r.exit_code == 0 and "Warning: no video found for x.expected.json" in r.output, r.output
