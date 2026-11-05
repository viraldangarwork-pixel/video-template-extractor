from pathlib import Path

import typer

from vtpl.pipeline import STAGE_NAMES
from vtpl.pipeline import analyze as run_analyze
from vtpl.render.run import render as run_render
from vtpl.render.slots import MediaError
from vtpl.schema import TemplateError
from vtpl.util.ffmpeg import FFmpegError, require_ffmpeg

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Video template extractor")


def _fail(msg: str) -> None:
    typer.echo(f"Error: {msg}")
    raise typer.Exit(1)


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
    typer.echo(f"Template written: {template}")


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
    except (FFmpegError, TemplateError, MediaError) as e:
        _fail(str(e))
    for w in warnings:
        typer.echo(f"Warning: {w}")
    typer.echo(f"Rendered: {out}")
