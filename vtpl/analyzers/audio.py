import bisect
from pathlib import Path

import numpy as np

from vtpl.util.workdir import WorkDir

OUTPUT = "audio.json"
HOP = 256  # finer than librosa default 512: tempo is quantised by frame hop
EMPTY = {"bpm": None, "beats_ms": [], "downbeat_hint_ms": []}


def analyze_beats(y: np.ndarray, sr: int) -> dict:
    import librosa

    if float(np.abs(y).max(initial=0.0)) < 1e-4:
        return dict(EMPTY)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0 or len(beat_frames) == 0:
        return dict(EMPTY)
    beats_ms = [int(round(t * 1000)) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)]
    strengths = onset_env[np.minimum(beat_frames, len(onset_env) - 1)]
    phase = max(range(4), key=lambda p: float(strengths[p::4].sum()))
    return {"bpm": round(bpm, 2), "beats_ms": beats_ms, "downbeat_hint_ms": beats_ms[phase::4]}


def is_on_beat(ms: int, beats_ms: list[int], tol_ms: int = 80) -> bool:
    if not beats_ms:
        return False
    i = bisect.bisect_left(beats_ms, ms)
    near = [beats_ms[j] for j in (i - 1, i) if 0 <= j < len(beats_ms)]
    return min(abs(ms - b) for b in near) <= tol_ms


def beat_pattern(shots: list[dict], period_ms: float) -> list[int]:
    return [max(1, int(round((s["end_ms"] - s["start_ms"]) / period_ms))) for s in shots]


def beats_of_file(path: Path) -> dict:
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    return analyze_beats(y, sr)


def run(wd: WorkDir, source: Path) -> list[str]:
    if not wd.read_json("meta.json")["has_audio"]:
        wd.write_json(OUTPUT, dict(EMPTY))
        return ["no audio: beats skipped"]
    wd.write_json(OUTPUT, beats_of_file(wd.path("audio.wav")))
    return []
