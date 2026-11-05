from pathlib import Path

from vtpl.util.ffmpeg import duration_ms, probe, run_ffmpeg
from vtpl.util.workdir import WorkDir

OUTPUT = "meta.json"
W, H, FPS, SR = 1080, 1920, 30, 22050

_VF = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
       f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p")


def run(wd: WorkDir, source: Path) -> list[str]:
    streams = probe(source)["streams"]
    if not any(s["codec_type"] == "video" for s in streams):
        raise ValueError(f"{source} has no video stream")
    has_audio = any(s["codec_type"] == "audio" for s in streams)

    run_ffmpeg(["-i", str(source), "-vf", _VF, "-an", "-c:v", "libx264", "-crf", "18",
                "-preset", "veryfast", str(wd.path("norm.mp4"))])
    if has_audio:
        run_ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar", str(SR), str(wd.path("audio.wav"))])

    wd.write_json(OUTPUT, {
        "w": W, "h": H, "fps": FPS,
        "duration_ms": duration_ms(wd.path("norm.mp4")),
        "has_audio": has_audio,
        "source": str(source),
    })
    return [] if has_audio else ["no audio stream"]
