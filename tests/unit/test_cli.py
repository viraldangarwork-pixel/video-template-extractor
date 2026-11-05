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
    _clean_failure(r, "Error:")


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
