import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegError(f"{tool} not found on PATH. Install FFmpeg 6 or newer (e.g. `sudo apt install ffmpeg`).")
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if " subtitles " not in filters:
        raise FFmpegError("Your FFmpeg has no 'subtitles' filter (built without libass). Install a full FFmpeg build.")


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed: {proc.stderr.strip()}\ncommand: {' '.join(cmd)}")


def probe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe could not read {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def duration_ms(path: Path) -> int:
    return int(round(float(probe(path)["format"]["duration"]) * 1000))
