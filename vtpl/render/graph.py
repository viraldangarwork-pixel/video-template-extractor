from dataclasses import dataclass
from pathlib import Path

XFADE = {"dissolve": "fade", "fade": "fadeblack", "flash": "fadewhite"}
W, H, FPS = 1080, 1920, 30


@dataclass(frozen=True)
class Segment:
    slot: str
    length_frames: int    # shot length plus transition overlaps, in output frames
    trans_in: str         # cut | fade | dissolve | flash
    trans_in_frames: int
    offset_frames: int    # where this segment starts in the output timeline (xfade offset)


def frames(ms: int) -> int:
    """Map a template time to the nearest output frame index, so all boundaries share one grid."""
    return round(ms * FPS / 1000)


def frames_to_ms(n: int) -> int:
    return -(-n * 1000 // FPS)    # ceil, so a clip trimmed to this many ms still covers n frames


def _dur(tin: dict) -> int:
    return 0 if tin["type"] == "cut" else frames(tin["duration_ms"])


def plan_segments(template: dict) -> list[Segment]:
    shots = template["shots"]
    segs = []
    for i, s in enumerate(shots):
        d_in = _dur(s["transition_in"]) if i > 0 else 0
        d_out = _dur(shots[i + 1]["transition_in"]) if i + 1 < len(shots) else 0
        f0, f1 = frames(s["start_ms"]), frames(s["end_ms"])
        length = (f1 - f0) + d_in // 2 + (d_out - d_out // 2)
        segs.append(Segment(s["slot"], length, s["transition_in"]["type"] if d_in else "cut", d_in,
                            f0 - d_in // 2))
    return segs


def trim_start_ms(clip_ms: int, need_ms: int) -> int:
    return max(0, (clip_ms - need_ms) // 2)


def _s(ms: int) -> str:
    return f"{ms / 1000:.3f}"


def _f(n: int) -> str:
    return f"{n / FPS:.6f}"


def build_filter(segs: list[Segment], has_music: bool, duration_ms: int) -> str:
    parts = [
        f"[{i}:v]setpts=PTS-STARTPTS,"
        f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={FPS},"
        f"trim=end_frame={seg.length_frames},setpts=PTS-STARTPTS,format=yuv420p,settb=AVTB[s{i}]"
        for i, seg in enumerate(segs)
    ]
    cur = "[s0]"
    for i in range(1, len(segs)):
        seg, nxt = segs[i], f"[j{i}]"
        if seg.trans_in == "cut":
            parts.append(f"{cur}[s{i}]concat=n=2:v=1:a=0{nxt}")
        else:
            parts.append(f"{cur}[s{i}]xfade=transition={XFADE[seg.trans_in]}:"
                         f"duration={_f(seg.trans_in_frames)}:offset={_f(seg.offset_frames)}{nxt}")
        cur = nxt
    parts.append(f"{cur}trim=duration={_s(duration_ms)},setpts=PTS-STARTPTS[vout]")
    if has_music:
        n = len(segs)
        fade_at = max(0, duration_ms - 300)
        parts.append(f"[{n}:a]atrim=0:{_s(duration_ms)},asetpts=PTS-STARTPTS,"
                     f"afade=t=out:st={_s(fade_at)}:d=0.3[aout]")
    return ";".join(parts)


def build_args(segs: list[Segment], clips: list[Path], trims_ms: list[int], music: Path | None,
               duration_ms: int, out: Path) -> list[str]:
    args: list[str] = []
    for p, t0 in zip(clips, trims_ms):
        args += ["-ss", _s(t0), "-stream_loop", "-1", "-i", str(p)]    # input seek: no decoding up to t0
    if music is not None:
        args += ["-stream_loop", "-1", "-i", str(music)]
    args += ["-filter_complex", build_filter(segs, music is not None, duration_ms), "-map", "[vout]"]
    if music is not None:
        args += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    args += ["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-t", _s(duration_ms), str(out)]
    return args
