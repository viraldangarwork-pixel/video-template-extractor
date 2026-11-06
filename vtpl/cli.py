import json
from pathlib import Path
from typing import NoReturn

import typer

from vtpl.pipeline import STAGE_NAMES
from vtpl.pipeline import analyze as run_analyze
from vtpl.render.run import render as run_render
from vtpl.render.slots import MediaError
from vtpl.schema import TemplateError
from vtpl.util.ffmpeg import FFmpegError, require_ffmpeg

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Video template extractor")


def _fail(msg: str) -> NoReturn:
    typer.echo(f"Error: {msg}")
    raise typer.Exit(1)


def _os_message(e: OSError) -> str:
    return f"{e.strerror}: {e.filename}" if e.strerror and e.filename else str(e)


@app.command()
def analyze(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source video"),
    out: Path = typer.Option(Path("work"), help="Work folder root"),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if cached"),
    stages: str = typer.Option("", help=f"Comma-separated subset of: {','.join(STAGE_NAMES)}"),
):
    """Analyse VIDEO and write template.json."""
    only = {s.strip() for s in stages.split(",") if s.strip()} or None
    unknown = sorted((only or set()) - set(STAGE_NAMES))
    if unknown:
        _fail(f"unknown stages: {', '.join(unknown)}. Valid stages: {', '.join(STAGE_NAMES)}")
    try:
        require_ffmpeg()
        template = run_analyze(video, out, force, only)
    except (FFmpegError, TemplateError, ValueError) as e:
        _fail(str(e))
    except OSError as e:
        msg = _os_message(e)
        if isinstance(e, FileNotFoundError) and e.filename and Path(e.filename).name == "meta.json":
            msg += " (run the ingest stage first)"
        _fail(msg)
    run_json = template.parent / "run.json"
    if run_json.exists():
        for stage, info in json.loads(run_json.read_text()).get("stages", {}).items():
            if info.get("status") == "failed":
                typer.echo(f"Warning: {stage} failed: {(info.get('warnings') or ['unknown error'])[0]}")
    if template.exists():
        typer.echo(f"Template written: {template}")
    else:
        typer.echo("Analysis done; template.json not written (merge stage not run)")


@app.command()
def render(
    template: Path = typer.Argument(..., exists=True, dir_okay=False, help="template.json"),
    media: Path = typer.Option(..., exists=True, file_okay=False, help="Folder with clip_N files, music, texts.yaml"),
    out: Path = typer.Option(Path("out.mp4"), help="Output MP4"),
    snap_to_beats: bool = typer.Option(False, "--snap-to-beats", help="Re-time shots to your music's beats"),
):
    """Render TEMPLATE with your own media."""
    try:
        warnings = run_render(template, media, out, snap_to_beats=snap_to_beats)
    except (FFmpegError, TemplateError, MediaError, ValueError) as e:
        _fail(str(e))
    except OSError as e:
        _fail(_os_message(e))
    for w in warnings:
        typer.echo(f"Warning: {w}")
    typer.echo(f"Rendered: {out}")


@app.command("eval")
def eval_cmd(
    golden: Path = typer.Option(..., exists=True, file_okay=False, help="Folder of videos + <name>.expected.json"),
    out: Path = typer.Option(Path("work"), help="Work folder root (eval_report.json is written here)"),
    stages: str = typer.Option("", help=f"Comma-separated subset of: {','.join(STAGE_NAMES)}"),
):
    """Score analysis against hand-checked expected results."""
    from vtpl.eval import evaluate, format_report
    only = {s.strip() for s in stages.split(",") if s.strip()} or None
    unknown = sorted((only or set()) - set(STAGE_NAMES))
    if unknown:
        _fail(f"unknown stages: {', '.join(unknown)}. Valid stages: {', '.join(STAGE_NAMES)}")
    try:
        require_ffmpeg()
        result = evaluate(golden, out, only)
    except (FFmpegError, TemplateError, ValueError, OSError) as e:
        _fail(str(e))
    typer.echo(format_report(result["videos"], result["summary"]))
