from typer.testing import CliRunner

from vtpl import cli

runner = CliRunner()


def test_help_lists_commands():
    r = runner.invoke(cli.app, ["--help"])
    assert r.exit_code == 0 and "analyze" in r.output and "render" in r.output


def test_unknown_stage_rejected(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    r = runner.invoke(cli.app, ["analyze", str(v), "--stages", "shots,bogus"])
    assert r.exit_code == 1 and "unknown stages: bogus" in r.output


def test_analyze_calls_pipeline(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "run_analyze", lambda src, out, force, only: seen.update(
        src=src, force=force, only=only) or tmp_path / "template.json")
    r = runner.invoke(cli.app, ["analyze", str(v), "--force", "--stages", "ingest,shots"])
    assert r.exit_code == 0 and "template.json" in r.output
    assert seen == {"src": v, "force": True, "only": {"ingest", "shots"}}


def test_render_reports_media_error(tmp_path, monkeypatch):
    from vtpl.render.slots import MediaError
    tpl = tmp_path / "t.json"
    tpl.write_text("{}")

    def boom(*a, **k):
        raise MediaError("No clip for clip_1")
    monkeypatch.setattr(cli, "run_render", boom)
    r = runner.invoke(cli.app, ["render", str(tpl), "--media", str(tmp_path)])
    assert r.exit_code == 1 and "No clip for clip_1" in r.output


def _clean_failure(r, *needles):
    assert r.exit_code == 1, r.output
    assert isinstance(r.exception, SystemExit), repr(r.exception)      # not an uncaught exception
    assert "Traceback" not in r.output
    for n in needles:
        assert n in r.output, r.output


def test_render_reports_invalid_json(tmp_path):
    tpl = tmp_path / "t.json"
    tpl.write_text("{not json")
    r = runner.invoke(cli.app, ["render", str(tpl), "--media", str(tmp_path)])
    _clean_failure(r, "Error: t.json is not valid JSON: Expecting property name")


def test_render_reports_os_error(tmp_path, monkeypatch):
    tpl = tmp_path / "t.json"
    tpl.write_text("{}")

    def boom(*a, **k):
        raise PermissionError(13, "Permission denied", str(tmp_path / "out.mp4"))
    monkeypatch.setattr(cli, "run_render", boom)
    r = runner.invoke(cli.app, ["render", str(tpl), "--media", str(tmp_path)])
    _clean_failure(r, "Error:", "Permission denied")


def test_analyze_missing_meta_hints_ingest(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    r = runner.invoke(cli.app, ["analyze", str(v), "--out", str(tmp_path / "work"), "--stages", "shots"])
    _clean_failure(r, "Error:", "meta.json", "run the ingest stage first")


def test_analyze_reports_other_os_error_without_ingest_hint(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)

    def boom(*a, **k):
        raise PermissionError(13, "Permission denied", str(tmp_path / "work"))
    monkeypatch.setattr(cli, "run_analyze", boom)
    r = runner.invoke(cli.app, ["analyze", str(v)])
    _clean_failure(r, "Error:", "Permission denied")
    assert "ingest" not in r.output


def test_analyze_without_merge_does_not_claim_template(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "run_analyze", lambda *a: tmp_path / "missing" / "template.json")
    r = runner.invoke(cli.app, ["analyze", str(v), "--stages", "ingest"])
    assert r.exit_code == 0
    assert "Analysis done; template.json not written (merge stage not run)" in r.output
    assert "Template written" not in r.output


def test_analyze_reports_written_template(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    tpl = tmp_path / "template.json"
    tpl.write_text("{}")
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "run_analyze", lambda *a: tpl)
    r = runner.invoke(cli.app, ["analyze", str(v)])
    assert r.exit_code == 0 and f"Template written: {tpl}" in r.output


def test_fail_is_annotated_noreturn():
    import typing
    assert typing.get_type_hints(cli._fail)["return"] is typing.NoReturn


def test_analyze_ingest_only_on_fresh_dir(tmp_path, monkeypatch):
    from vtpl import pipeline
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")

    def stage(name, output, needs=()):
        def fn(wd, source):
            for n in needs:
                wd.read_json(n)                                         # like the real merge
            wd.write_json(output, {})
            return []
        return pipeline.Stage(name, output, fn)
    monkeypatch.setattr(pipeline, "STAGES", [stage("ingest", "meta.json"), stage("shots", "shots.json"),
                                             stage("merge", "template.json", ("meta.json", "shots.json"))])
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    r = runner.invoke(cli.app, ["analyze", str(v), "--out", str(tmp_path / "work"), "--stages", "ingest"])
    assert r.exit_code == 0, r.output
    assert "Analysis done; template.json not written (merge stage not run)" in r.output


def test_analyze_reports_failed_optional_stages(tmp_path, monkeypatch):
    import json
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    work = tmp_path / "work" / "abc"
    work.mkdir(parents=True)

    def fake(src, out, force, only):
        (work / "run.json").write_text(json.dumps({"stages": {
            "ingest": {"status": "ok", "seconds": 0.1, "warnings": []},
            "speech": {"status": "failed", "seconds": 0.1, "warnings": ["model missing", "second"]}}}))
        return work / "template.json"

    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "run_analyze", fake)
    r = runner.invoke(cli.app, ["analyze", str(v)])
    assert r.exit_code == 0
    assert "Warning: speech failed: model missing" in r.output
    assert "ingest failed" not in r.output and "second" not in r.output


def test_render_reports_unknown_font(tmp_path, monkeypatch):
    import copy
    import json

    from tests.unit.test_schema import GOOD, TEXT
    from vtpl.render import run
    monkeypatch.setattr(run, "duration_ms", lambda p: 5000)      # dummy clips: fail on the font, not on probing
    for slot in ("clip_1", "clip_2"):
        (tmp_path / f"{slot}.mp4").write_bytes(b"x")
    t = copy.deepcopy(GOOD)
    t["texts"] = [copy.deepcopy(TEXT)]
    t["texts"][0]["style"]["font"] = "Comic-Sans"
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps(t))
    r = runner.invoke(cli.app, ["render", str(tpl), "--media", str(tmp_path)])
    _clean_failure(r, "Error:", "texts/0/style/font: unknown font 'Comic-Sans'", "catalog.json")
