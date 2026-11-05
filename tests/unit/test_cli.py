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
