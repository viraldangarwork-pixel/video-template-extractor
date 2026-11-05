import json
import shlex
from pathlib import Path

from vtpl.analyzers.audio import beats_of_file
from vtpl.render.graph import build_args, frames_to_ms, plan_segments, trim_start_ms
from vtpl.render.slots import MediaError, resolve_media
from vtpl.render.snap import beat_phase_ms, snap_template
from vtpl.schema import TemplateError, validate
from vtpl.util.ffmpeg import duration_ms, require_ffmpeg, run_ffmpeg


def render(template_path: Path, media_dir: Path, out: Path, snap_to_beats: bool = False) -> list[str]:
    require_ffmpeg()
    try:
        template = json.loads(Path(template_path).read_text())
    except json.JSONDecodeError as e:
        raise TemplateError(f"{Path(template_path).name} is not valid JSON: {e}") from None
    validate(template)
    media = resolve_media(template, media_dir)
    if snap_to_beats:
        if media.music is None:
            raise MediaError("--snap-to-beats needs a music file in the media folder")
        beats = beats_of_file(media.music)
        if beats["bpm"] is None:
            raise MediaError(f"No beats found in {media.music.name}; render without --snap-to-beats")
        phase = beat_phase_ms(beats["beats_ms"], 60000.0 / beats["bpm"])
        template = snap_template(template, beats["bpm"], first_beat_ms=phase)
        validate(template)

    segs = plan_segments(template)
    clips = [media.clips[s.slot] for s in segs]
    trims = [trim_start_ms(duration_ms(c), frames_to_ms(s.length_frames)) for c, s in zip(clips, segs)]
    args = build_args(segs, clips, trims, media.music, template["meta"]["duration_ms"], Path(out))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    (Path(out).parent / "render_cmd.txt").write_text("ffmpeg " + shlex.join(args) + "\n")
    run_ffmpeg(args)
    return media.warnings
