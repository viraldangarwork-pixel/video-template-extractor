"""Synthetic test media with known ground truth. All videos are 30 fps."""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FPS = 30
CARDS = [("red", 1.5), ("blue", 1.0), ("green", 1.5), ("purple", 2.0)]


def _ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def click_signal(seconds: float, bpm: float = 120.0, sr: int = 22050, first_s: float = 0.5):
    """Return (samples, sr, beat_times_ms) for a click every beat starting at first_s."""
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    t = np.arange(int(0.02 * sr)) / sr
    burst = (0.8 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 200)).astype(np.float32)
    beats, s = [], first_s
    while s + 0.02 < seconds:
        i = int(round(s * sr))
        y[i:i + len(burst)] += burst
        beats.append(int(round(s * 1000)))
        s += 60.0 / bpm
    return y, sr, beats


def click_track(path: Path, seconds: float, bpm: float = 120.0) -> list[int]:
    y, sr, beats = click_signal(seconds, bpm)
    sf.write(path, y, sr)
    return beats


def color_cards(path: Path, cards: list[tuple[str, float]], audio: Path | None = None,
                size: str = "1080x1920") -> list[int]:
    """Hard-cut solid colour cards. Returns cut times in ms."""
    args, filt = [], ""
    for i, (color, dur) in enumerate(cards):
        args += ["-f", "lavfi", "-i", f"color=c={color}:s={size}:r={FPS}:d={dur}"]
        filt += f"[{i}:v]"
    filt += f"concat=n={len(cards)}:v=1:a=0,format=yuv420p[v]"
    maps = ["-map", "[v]"]
    if audio is not None:
        args += ["-i", str(audio)]
        maps += ["-map", f"{len(cards)}:a", "-c:a", "aac", "-shortest"]
    _ff([*args, "-filter_complex", filt, *maps, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(path)])
    cuts, t = [], 0.0
    for _, dur in cards[:-1]:
        t += dur
        cuts.append(int(round(t * 1000)))
    return cuts


def transitions(path: Path) -> list[dict]:
    """red -dissolve-> blue -fade(black)-> green -flash(white)-> purple; each card 2 s."""
    specs = [("fade", 0.5, "dissolve"), ("fadeblack", 0.5, "fade"), ("fadewhite", 0.2, "flash")]
    colors = ["red", "blue", "green", "purple"]
    args: list[str] = []
    for c in colors:
        args += ["-f", "lavfi", "-i", f"color=c={c}:s=1080x1920:r={FPS}:d=2"]
    filt, prev, length, truth = "", "[0:v]", 2.0, []
    for i, (xf, d, kind) in enumerate(specs, start=1):
        off = round(length - d, 3)
        filt += f"{prev}[{i}:v]xfade=transition={xf}:duration={d}:offset={off}[x{i}];"
        truth.append({"ms": int(round((off + d / 2) * 1000)), "type": kind, "duration_ms": int(round(d * 1000))})
        prev, length = f"[x{i}]", off + 2.0
    filt += f"{prev}format=yuv420p[v]"
    _ff([*args, "-filter_complex", filt, "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(path)])
    return truth


def solid_clip(path: Path, color: str, seconds: float = 3.0) -> None:
    _ff(["-f", "lavfi", "-i", f"color=c={color}:s=1080x1920:r={FPS}:d={seconds}",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)])
