"""vtpl — Video Template Extractor & Renderer SDK.

Python package for analyzing short-form videos (Reels, TikTok, Shorts),
extracting editable JSON templates, and re-rendering new videos.
"""
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from vtpl.pipeline import analyze as _analyze_pipeline
from vtpl.render.run import render as _render_pipeline
from vtpl.render.slots import MediaError
from vtpl.schema import TemplateError, validate as validate_template
from vtpl.util.ffmpeg import FFmpegError

__version__ = "0.1.0"
__all__ = [
    "analyze",
    "render",
    "load_template",
    "save_template",
    "validate_template",
    "serve",
    "TemplateError",
    "MediaError",
    "FFmpegError",
    "__version__",
]


def analyze(
    video: Union[str, Path],
    out_dir: Union[str, Path] = "work",
    force: bool = False,
    stages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze a short-form video and extract an editable template dictionary.

    Args:
        video: Path to input video (.mp4, .mov, .webm, etc.).
        out_dir: Work directory root for caching and stage outputs (default: 'work').
        force: Whether to re-run stages even if cached outputs exist.
        stages: Optional list of specific stages to run (e.g. ['ingest', 'shots', 'audio', 'merge']).

    Returns:
        The extracted template as a Python dictionary.
    """
    video_path = Path(video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    only_stages = set(stages) if stages else None
    template_path = _analyze_pipeline(video_path, Path(out_dir).resolve(), force=force, only=only_stages)
    if not template_path.exists():
        raise RuntimeError(f"Analysis completed but {template_path} was not created.")
    return json.loads(template_path.read_text(encoding="utf-8"))


def render(
    template: Union[str, Path, Dict[str, Any]],
    media: Union[str, Path],
    out: Union[str, Path] = "out.mp4",
    snap_to_beats: bool = False,
) -> List[str]:
    """Render a new video from a template and a media directory.

    Args:
        template: Path to template.json OR a template dictionary.
        media: Directory containing replacement clips (clip_1.mp4 ...), music, and optional texts.yaml.
        out: Path for the destination MP4 video file.
        snap_to_beats: Re-time shots to match the new music's rhythm.

    Returns:
        List of any non-fatal warning messages encountered during rendering.
    """
    media_dir = Path(media).resolve()
    out_path = Path(out).resolve()

    if isinstance(template, dict):
        validate_template(template)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write(json.dumps(template, indent=2))
            tmp_path = Path(tmp.name)
        try:
            return _render_pipeline(tmp_path, media_dir, out_path, snap_to_beats=snap_to_beats)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        tmpl_path = Path(template).resolve()
        if not tmpl_path.exists():
            raise FileNotFoundError(f"Template file not found: {tmpl_path}")
        return _render_pipeline(tmpl_path, media_dir, out_path, snap_to_beats=snap_to_beats)


def load_template(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a template.json file from disk."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Template file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    validate_template(data)
    return data


def save_template(template: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> Path:
    """Validate and save a template dictionary to a JSON file."""
    validate_template(template)
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(template, indent=indent), encoding="utf-8")
    return p


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Launch the built-in web studio application programmatically."""
    import uvicorn
    uvicorn.run("vtpl.web.app:app", host=host, port=port, reload=reload)
