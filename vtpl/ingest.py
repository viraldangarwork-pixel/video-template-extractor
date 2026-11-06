from pathlib import Path

from vtpl.util.ffmpeg import duration_ms, probe, run_ffmpeg
from vtpl.util.workdir import WorkDir

OUTPUT = "meta.json"
W, H, FPS, SR = 1080, 1920, 30, 22050

_VF = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
       f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p")


def content_rect(src_w: int, src_h: int, rotation: float = 0) -> dict:
    """Where the source picture lands in the normalised frame (the rest is black padding)."""
    if round(abs(rotation)) % 180 == 90:       # FFmpeg auto-rotates, so the displayed frame is transposed
        src_w, src_h = src_h, src_w
    if src_w * H >= src_h * W:
        w, h = W, min(H, round(W * src_h / src_w))
    else:
        w, h = min(W, round(H * src_w / src_h)), H
    w, h = max(2, w - w % 2), max(2, h - h % 2)
    x, y = (W - w) // 2, (H - h) // 2
    return {"x": x - x % 2, "y": y - y % 2, "w": w, "h": h}


def _rotation(stream: dict) -> float:
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            return float(sd["rotation"])
    return float(stream.get("tags", {}).get("rotate", 0))


def run(wd: WorkDir, source: Path) -> list[str]:
    streams = probe(source)["streams"]
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if video is None:
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
        "content": content_rect(int(video["width"]), int(video["height"]), _rotation(video)),
    })
    return [] if has_audio else ["no audio stream"]
