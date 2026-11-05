# vtpl Plan 1 — Core Pipeline (shots, transitions, beats, speech, template, render) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `vtpl analyze` → `template.json` → `vtpl render` loop for cuts, transitions, beats and speech words, verified by a synthetic round-trip test.

**Architecture:** Independent pipeline stages (pure functions: files in → one JSON out) under `work/<video_id>/`, cached unless `--force`. `merge` builds a schema-validated `template.json`; the renderer turns it plus the user's media into one FFmpeg `filter_complex` command. All times are integer milliseconds.

**Tech Stack:** Python 3.11+, FFmpeg 6 (libass), PySceneDetect, OpenCV (headless), NumPy, librosa, soundfile, faster-whisper, jsonschema, PyYAML, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-video-template-extractor-design.md`

**Plan series (each produces working software):**
- **Plan 1 (this file):** project skeleton, ingest, shots + transitions, beats, speech, schema, merge, pipeline, renderer (cuts/xfade/music/snap-to-beats), CLI, round-trip tests.
- **Plan 2:** frame cache, text detect/track/refine/style/font/animation, caption-vs-overlay dedupe, ASS text + karaoke caption rendering.
- **Plan 3:** camera motion + colour grade (analysis and `zoompan`/`eq` rendering), `vtpl eval` golden set.

---

## File structure (Plan 1)

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, `vtpl` entry point, pytest config |
| `vtpl/__init__.py` | Version |
| `vtpl/util/timebase.py` | ms ↔ frame conversion |
| `vtpl/util/workdir.py` | `video_id`, `WorkDir` (paths, JSON IO, `run.json` + `<stage>.log`) |
| `vtpl/util/ffmpeg.py` | `require_ffmpeg`, `run_ffmpeg`, `probe`, `duration_ms`, `FFmpegError` |
| `vtpl/ingest.py` | Normalise video to 1080×1920 30 fps, extract mono WAV, write `meta.json` |
| `vtpl/analyzers/framesig.py` | Decode tiny RGB frames + per-frame luma (signal for shot analysis) |
| `vtpl/analyzers/transitions.py` | Pure classifiers: fades, flashes, dissolves, boundary merging |
| `vtpl/analyzers/shots.py` | PySceneDetect cuts + classification → `shots.json` |
| `vtpl/analyzers/audio.py` | Beats/BPM/downbeat hint, `is_on_beat`, `beat_pattern` → `audio.json` |
| `vtpl/analyzers/speech.py` | faster-whisper words → `speech.json` |
| `vtpl/timeline.py` | `clamp_transitions` shared by merge and snap |
| `vtpl/schema.py` | JSON Schema + timeline checks, `TemplateError`, `validate` |
| `vtpl/merge.py` | Build `template.json` from stage outputs |
| `vtpl/pipeline.py` | Stage registry, caching, failure policy |
| `vtpl/render/slots.py` | Resolve user media (clips, music, texts.yaml) |
| `vtpl/render/graph.py` | Pure FFmpeg filter-graph and argument builder |
| `vtpl/render/snap.py` | `--snap-to-beats` time warp |
| `vtpl/render/run.py` | Orchestrate a render |
| `vtpl/cli.py` | Typer CLI: `analyze`, `render` |
| `tests/synth/make_fixtures.py` | Generate synthetic videos/audio with ground truth |
| `tests/conftest.py` | Session fixtures (generated once per test run) |
| `tests/unit/*.py`, `tests/integration/*.py`, `tests/roundtrip/*.py` | Tests |

---

### Task 1: Project skeleton, timebase, workdir

**Files:**
- Create: `pyproject.toml`, `vtpl/__init__.py`, `vtpl/util/__init__.py`, `vtpl/util/timebase.py`, `vtpl/util/workdir.py`, `tests/__init__.py`, `tests/unit/__init__.py`
- Test: `tests/unit/test_timebase_workdir.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "vtpl"
version = "0.1.0"
description = "Reverse-engineer short-form videos into editable templates and re-render them."
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "opencv-python-headless>=4.9",
  "scenedetect>=0.6.4",
  "librosa>=0.10.2",
  "soundfile>=0.12",
  "faster-whisper>=1.0",
  "jsonschema>=4.21",
  "pyyaml>=6.0",
  "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
vtpl = "vtpl.cli:app"

[tool.setuptools.packages.find]
include = ["vtpl*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = ["slow: runs FFmpeg on generated videos"]
```

- [ ] **Step 2: Create package files**

`vtpl/__init__.py`:
```python
__version__ = "0.1.0"
```
`vtpl/util/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`: empty files.

- [ ] **Step 3: Install**

Run:
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
```
Expected: ends with `Successfully installed ... vtpl-0.1.0`.

- [ ] **Step 4: Write the failing test** — `tests/unit/test_timebase_workdir.py`

```python
from pathlib import Path

from vtpl.util.timebase import frame_to_ms, ms_to_frame
from vtpl.util.workdir import WorkDir, video_id


def test_frame_ms_round_trip():
    assert frame_to_ms(45) == 1500
    assert frame_to_ms(1) == 33
    assert ms_to_frame(1500) == 45
    assert ms_to_frame(1517) == 46
    assert frame_to_ms(15, fps=15) == 1000


def test_video_id_is_stable_12_hex(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    vid = video_id(f)
    assert vid == video_id(f)
    assert len(vid) == 12 and all(c in "0123456789abcdef" for c in vid)


def test_workdir_json_and_record(tmp_path: Path):
    wd = WorkDir(tmp_path / "w")
    assert not wd.has("x.json")
    wd.write_json("x.json", {"a": 1})
    assert wd.has("x.json") and wd.read_json("x.json") == {"a": 1}
    wd.record("shots", "ok", 1.234, ["minor"])
    wd.record("audio", "failed", 0.5, ["boom"])
    run = wd.read_json("run.json")
    assert run["stages"]["shots"] == {"status": "ok", "seconds": 1.23, "warnings": ["minor"]}
    assert run["stages"]["audio"]["status"] == "failed"
    log = wd.path("audio.log").read_text()
    assert "status: failed" in log and "warning: boom" in log
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/unit/test_timebase_workdir.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.util.timebase'`

- [ ] **Step 6: Implement** — `vtpl/util/timebase.py`

```python
FPS = 30


def frame_to_ms(frame: int, fps: float = FPS) -> int:
    return int(round(frame * 1000 / fps))


def ms_to_frame(ms: int, fps: float = FPS) -> int:
    return int(round(ms * fps / 1000))
```

`vtpl/util/workdir.py`:
```python
import hashlib
import json
from pathlib import Path
from typing import Any


def video_id(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


class WorkDir:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def write_json(self, name: str, data: Any) -> None:
        self.path(name).write_text(json.dumps(data, indent=2))

    def read_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text())

    def record(self, stage: str, status: str, seconds: float, warnings: list[str]) -> None:
        run = self.read_json("run.json") if self.has("run.json") else {"stages": {}}
        run["stages"][stage] = {"status": status, "seconds": round(seconds, 2), "warnings": list(warnings)}
        self.write_json("run.json", run)
        lines = [f"status: {status}", f"seconds: {seconds:.2f}", *[f"warning: {w}" for w in warnings]]
        self.path(f"{stage}.log").write_text("\n".join(lines) + "\n")
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/unit/test_timebase_workdir.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml vtpl tests
git commit -m "feat: project skeleton, timebase and workdir helpers"
```

---

### Task 2: FFmpeg utilities

**Files:**
- Create: `vtpl/util/ffmpeg.py`
- Test: `tests/unit/test_ffmpeg_util.py`

- [ ] **Step 1: Write the failing test**

```python
import shutil
from pathlib import Path

import pytest

from vtpl.util import ffmpeg as ff

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_require_ffmpeg_passes_when_installed():
    ff.require_ffmpeg()


def test_require_ffmpeg_reports_missing_tool(monkeypatch):
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    with pytest.raises(ff.FFmpegError, match="ffmpeg not found on PATH"):
        ff.require_ffmpeg()


def test_run_and_probe(tmp_path: Path):
    out = tmp_path / "c.mp4"
    ff.run_ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=64x64:r=30:d=1", "-pix_fmt", "yuv420p", str(out)])
    info = ff.probe(out)
    assert info["streams"][0]["codec_type"] == "video"
    assert abs(ff.duration_ms(out) - 1000) <= 40


def test_run_ffmpeg_error_has_stderr(tmp_path: Path):
    with pytest.raises(ff.FFmpegError, match="ffmpeg failed"):
        ff.run_ffmpeg(["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "o.mp4")])


def test_probe_error(tmp_path: Path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ff.FFmpegError, match="ffprobe could not read"):
        ff.probe(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ffmpeg_util.py -v`
Expected: FAIL with `ImportError: cannot import name 'ffmpeg' from 'vtpl.util'`

- [ ] **Step 3: Implement** — `vtpl/util/ffmpeg.py`

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_ffmpeg_util.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/util/ffmpeg.py tests/unit/test_ffmpeg_util.py
git commit -m "feat: ffmpeg/ffprobe helpers with clear errors"
```

---

### Task 3: Synthetic fixture generator

**Files:**
- Create: `tests/synth/__init__.py` (empty), `tests/synth/make_fixtures.py`, `tests/conftest.py`
- Test: `tests/unit/test_fixtures.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_fixtures.py`

```python
import pytest

from tests.synth.make_fixtures import CARDS, click_signal
from vtpl.util.ffmpeg import duration_ms, probe


def test_click_signal_beats():
    y, sr, beats = click_signal(seconds=4.0, bpm=120.0)
    assert sr == 22050 and len(y) == 4 * 22050
    assert beats == [500, 1000, 1500, 2000, 2500, 3000, 3500]


@pytest.mark.slow
def test_cards_video(cards_video):
    assert cards_video["cuts_ms"] == [1500, 2500, 4000]
    assert abs(duration_ms(cards_video["path"]) - int(sum(d for _, d in CARDS) * 1000)) <= 50
    kinds = {s["codec_type"] for s in probe(cards_video["path"])["streams"]}
    assert kinds == {"video", "audio"}


@pytest.mark.slow
def test_transitions_video(transitions_video):
    assert [t["type"] for t in transitions_video["truth"]] == ["dissolve", "fade", "flash"]
    assert [t["ms"] for t in transitions_video["truth"]] == [1750, 3250, 4900]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.synth'`

- [ ] **Step 3: Implement** — `tests/synth/make_fixtures.py`

```python
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
```

`tests/conftest.py`:
```python
import shutil

import pytest

from tests.synth import make_fixtures as mf


def _need_ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def cards_video(fixtures_dir):
    _need_ffmpeg()
    seconds = sum(d for _, d in mf.CARDS)
    wav = fixtures_dir / "clicks.wav"
    beats = mf.click_track(wav, seconds=seconds)
    path = fixtures_dir / "cards.mp4"
    cuts = mf.color_cards(path, mf.CARDS, audio=wav)
    return {"path": path, "cuts_ms": cuts, "beats_ms": beats, "wav": wav}


@pytest.fixture(scope="session")
def small_cards_video(fixtures_dir):
    _need_ffmpeg()
    path = fixtures_dir / "small_cards.mp4"
    wav = fixtures_dir / "clicks_small.wav"
    mf.click_track(wav, seconds=6.0)
    mf.color_cards(path, mf.CARDS, audio=wav, size="540x960")
    return path


@pytest.fixture(scope="session")
def transitions_video(fixtures_dir):
    _need_ffmpeg()
    path = fixtures_dir / "transitions.mp4"
    return {"path": path, "truth": mf.transitions(path)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_fixtures.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/synth tests/conftest.py tests/unit/test_fixtures.py
git commit -m "test: synthetic video/audio fixtures with ground truth"
```

---

### Task 4: Ingest stage

**Files:**
- Create: `vtpl/ingest.py`, `tests/integration/__init__.py` (empty)
- Test: `tests/integration/test_ingest.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from vtpl import ingest
from vtpl.util.ffmpeg import probe
from vtpl.util.workdir import WorkDir

pytestmark = pytest.mark.slow


def test_ingest_normalises_and_extracts_audio(tmp_path, small_cards_video):
    wd = WorkDir(tmp_path / "w")
    warnings = ingest.run(wd, small_cards_video)
    assert warnings == []
    v = probe(wd.path("norm.mp4"))["streams"][0]
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v["r_frame_rate"] == "30/1"
    assert wd.has("audio.wav")
    meta = wd.read_json("meta.json")
    assert meta["has_audio"] is True and meta["fps"] == 30
    assert abs(meta["duration_ms"] - 6000) <= 50


def test_ingest_without_audio(tmp_path, transitions_video):
    wd = WorkDir(tmp_path / "w")
    assert ingest.run(wd, transitions_video["path"]) == ["no audio stream"]
    assert wd.read_json("meta.json")["has_audio"] is False
    assert not wd.has("audio.wav")


def test_ingest_rejects_non_video(tmp_path):
    wav = tmp_path / "a.wav"
    from tests.synth.make_fixtures import click_track
    click_track(wav, seconds=1.0)
    with pytest.raises(ValueError, match="has no video stream"):
        ingest.run(WorkDir(tmp_path / "w"), wav)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_ingest.py -v`
Expected: FAIL with `ImportError: cannot import name 'ingest' from 'vtpl'`

- [ ] **Step 3: Implement** — `vtpl/ingest.py`

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_ingest.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/ingest.py tests/integration
git commit -m "feat: ingest stage normalises video and extracts audio"
```

---

### Task 5: Frame signal (tiny frames + luma)

**Files:**
- Create: `vtpl/analyzers/__init__.py` (empty), `vtpl/analyzers/framesig.py`
- Test: `tests/unit/test_framesig.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from vtpl.analyzers.framesig import luma, read_small_frames


def test_luma_weights():
    frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
    frames[1, ..., 0] = 1.0  # pure red
    assert luma(frames).tolist() == pytest.approx([0.0, 0.299])


@pytest.mark.slow
def test_read_small_frames(cards_video):
    f = read_small_frames(cards_video["path"])
    assert f.shape[1:] == (64, 36, 3) and f.dtype == np.float32
    assert 178 <= f.shape[0] <= 182          # 6 s at 30 fps
    assert f[0, ..., 0].mean() > 0.8          # first card is red


def test_read_small_frames_bad_file(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError, match="could not decode frames"):
        read_small_frames(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_framesig.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/framesig.py`

```python
from pathlib import Path

import cv2
import numpy as np

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def read_small_frames(video: Path, size: tuple[int, int] = (36, 64)) -> np.ndarray:
    """Every frame, resized to size=(w, h), RGB float32 in 0..1. Shape (N, h, w, 3)."""
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    if not out:
        raise ValueError(f"could not decode frames from {video}")
    return np.stack(out)


def luma(frames: np.ndarray) -> np.ndarray:
    """Mean Rec.601 luma per frame, shape (N,)."""
    return (frames @ _LUMA).mean(axis=(1, 2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_framesig.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers tests/unit/test_framesig.py
git commit -m "feat: tiny-frame decoder and luma signal"
```

---

### Task 6: Transition classifiers (pure)

**Files:**
- Create: `vtpl/analyzers/transitions.py`
- Test: `tests/unit/test_transitions.py`

Background for the engineer: a *dissolve* frame F between pure frames A (before) and B (after) satisfies F ≈ α·A + (1−α)·B. We solve for α by least squares and accept the window when the residual is small, α falls monotonically, and at least 3 frames are true blends. Fades and flashes are found from the luma curve alone.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from vtpl.analyzers.framesig import luma
from vtpl.analyzers.transitions import (
    classify_boundaries, dissolve_span, find_fades, find_flashes, runs,
)

RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
BLUE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
GREEN = np.array([0.0, 0.5, 0.0], dtype=np.float32)


def solid(color, n):
    return np.broadcast_to(color, (n, 8, 4, 3)).astype(np.float32).copy()


def blend(a, b, n):
    alphas = np.linspace(1, 0, n + 2)[1:-1]
    return np.stack([np.broadcast_to(al * a + (1 - al) * b, (8, 4, 3)) for al in alphas]).astype(np.float32)


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]


def test_hard_cut_stays_cut():
    frames = np.concatenate([solid(RED, 30), solid(BLUE, 30)])
    b = classify_boundaries([30], frames, luma(frames))
    assert [(x["frame"], x["type"], x["duration_frames"]) for x in b] == [(30, "cut", 0)]


def test_dissolve_detected_from_detector_cut():
    frames = np.concatenate([solid(RED, 30), blend(RED, BLUE, 15), solid(BLUE, 30)])
    span = dissolve_span(frames, 37)
    assert span is not None and abs(span[0] - 30) <= 1 and abs(span[1] - 44) <= 1
    b = classify_boundaries([37], frames, luma(frames))
    assert len(b) == 1 and b[0]["type"] == "dissolve" and abs(b[0]["frame"] - 37) <= 1
    assert 13 <= b[0]["duration_frames"] <= 15


def test_dissolve_found_without_detector_cut():
    frames = np.concatenate([solid(RED, 30), blend(RED, BLUE, 15), solid(BLUE, 30)])
    b = classify_boundaries([], frames, luma(frames))
    assert [x["type"] for x in b] == ["dissolve"]


def test_fade_through_black():
    black = np.zeros(3, dtype=np.float32)
    frames = np.concatenate([solid(BLUE, 30), blend(BLUE, black, 7), solid(black, 2),
                             blend(black, GREEN, 7), solid(GREEN, 30)])
    fades = find_fades(luma(frames))
    assert len(fades) == 1 and abs(fades[0]["frame"] - 38) <= 1 and fades[0]["type"] == "fade"
    b = classify_boundaries([31, 45], frames, luma(frames))   # detector noise inside the fade
    assert [x["type"] for x in b] == ["fade"]


def test_video_starting_black_is_not_a_fade():
    black = np.zeros(3, dtype=np.float32)
    frames = np.concatenate([solid(black, 5), blend(black, BLUE, 6), solid(BLUE, 30)])
    assert find_fades(luma(frames)) == []


def test_flash_between_shots():
    white = np.ones(3, dtype=np.float32)
    frames = np.concatenate([solid(GREEN, 30), blend(GREEN, white, 2), solid(white, 2),
                             blend(white, BLUE, 2), solid(BLUE, 30)])
    fl = find_flashes(luma(frames), frames)
    assert len(fl) == 1 and fl[0]["type"] == "flash" and abs(fl[0]["frame"] - 33) <= 1


def test_flash_within_same_shot_ignored():
    white = np.ones(3, dtype=np.float32)
    frames = np.concatenate([solid(GREEN, 30), solid(white, 2), solid(GREEN, 30)])
    assert find_flashes(luma(frames), frames) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_transitions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.transitions'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/transitions.py`

```python
"""Pure transition classifiers. Frames: float32 (N, h, w, 3) in 0..1. Luma: (N,)."""
import numpy as np

DARK = 0.06          # luma below this is "black"
BRIGHT = 0.85        # luma above this is "white"
EPS = 1e-3
PAD = 3              # frames around a transition span treated as part of it


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _span(luma: np.ndarray, a: int, b: int, outward_brighter: bool) -> tuple[int, int]:
    """Grow [a, b) outwards while luma keeps moving away from the dip/peak. Inclusive result."""
    s, e = a, b - 1
    sign = 1 if outward_brighter else -1
    while s > 0 and sign * (luma[s - 1] - luma[s]) > EPS:
        s -= 1
    while e < len(luma) - 1 and sign * (luma[e + 1] - luma[e]) > EPS:
        e += 1
    return s, e


def _boundary(kind: str, s: int, e: int) -> dict:
    return {"frame": (s + e + 1) // 2, "type": kind, "duration_frames": e - s + 1, "span": (s, e)}


def find_fades(luma: np.ndarray, min_ramp: int = 3) -> list[dict]:
    out = []
    for a, b in runs(luma < DARK):
        if a == 0 or b == len(luma):
            continue                                  # starts or ends in black: not between shots
        s, e = _span(luma, a, b, outward_brighter=True)
        if (a - s) >= min_ramp or (e - (b - 1)) >= min_ramp:
            out.append(_boundary("fade", s, e))
    return out


def find_flashes(luma: np.ndarray, frames: np.ndarray, max_len: int = 4,
                 min_rise: float = 0.3, min_change: float = 0.1) -> list[dict]:
    out = []
    for a, b in runs(luma > BRIGHT):
        if b - a > max_len or a < PAD or b + PAD > len(luma):
            continue
        peak = float(luma[a:b].max())
        if luma[a - PAD] > peak - min_rise or luma[b + PAD - 1] > peak - min_rise:
            continue
        if float(np.abs(frames[a - PAD] - frames[b + PAD - 1]).mean()) < min_change:
            continue                                  # same picture before and after: in-shot flash
        s, e = _span(luma, a, b, outward_brighter=False)
        out.append(_boundary("flash", s, e))
    return out


def dissolve_span(frames: np.ndarray, c: int, max_half: int = 12, res_thr: float = 0.25,
                  min_blend: int = 3) -> tuple[int, int] | None:
    """Inclusive frame span of a linear blend around frame c, or None."""
    best = None
    for half in range(2, max_half + 1):
        a, b = c - half - 1, c + half
        if a < 0 or b >= len(frames):
            break
        A, B = frames[a], frames[b]
        d = A - B
        denom = float((d * d).sum())
        if denom < 1e-6:
            break
        alphas, ok = [], True
        for i in range(a + 1, b):
            F = frames[i]
            al = float(((F - B) * d).sum() / denom)
            res = float(np.linalg.norm(F - (al * A + (1 - al) * B)) / np.sqrt(denom))
            if res > res_thr:
                ok = False
                break
            alphas.append(al)
        if not ok:
            break
        monotonic = all(x >= y - 0.05 for x, y in zip(alphas, alphas[1:]))
        idx = [a + 1 + j for j, al in enumerate(alphas) if 0.05 < al < 0.95]
        if monotonic and len(idx) >= min_blend:
            best = (idx[0], idx[-1])
    return best


def long_range_diff(frames: np.ndarray, k: int = 6) -> np.ndarray:
    L = np.zeros(len(frames), dtype=np.float32)
    for i in range(k, len(frames) - k):
        L[i] = float(np.abs(frames[i - k] - frames[i + k]).mean())
    return L


def dissolve_candidates(frames: np.ndarray, k: int = 6, thr: float = 0.15) -> list[int]:
    L = long_range_diff(frames, k)
    return [i for i in range(1, len(L) - 1) if L[i] >= thr and L[i] >= L[i - 1] and L[i] > L[i + 1]]


def classify_boundaries(cut_frames: list[int], frames: np.ndarray, luma: np.ndarray) -> list[dict]:
    """Merge detector cuts with luma/blend analysis into one sorted boundary list."""
    out = find_fades(luma) + find_flashes(luma, frames)

    def covered(f: int) -> bool:
        return any(x["span"][0] - PAD <= f <= x["span"][1] + PAD for x in out)

    for c in sorted(set(cut_frames)):
        if covered(c):
            continue
        span = dissolve_span(frames, c)
        out.append(_boundary("dissolve", *span) if span else
                   {"frame": c, "type": "cut", "duration_frames": 0, "span": (c, c)})
    for c in dissolve_candidates(frames):
        if covered(c) or any(abs(c - x["frame"]) <= 6 for x in out):
            continue
        span = dissolve_span(frames, c)
        if span:
            out.append(_boundary("dissolve", *span))
    return sorted(out, key=lambda x: x["frame"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_transitions.py -v`
Expected: 8 passed. If a threshold test fails, print the luma/alpha arrays for that case and adjust only the named constant (`DARK`, `BRIGHT`, `res_thr`), never the test's ground truth.

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/transitions.py tests/unit/test_transitions.py
git commit -m "feat: fade/flash/dissolve transition classifiers"
```

---

### Task 7: Shots stage

**Files:**
- Create: `vtpl/analyzers/shots.py`
- Test: `tests/unit/test_build_shots.py`, `tests/integration/test_shots.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_build_shots.py`:
```python
from vtpl.analyzers.shots import build_shots


def test_build_shots_from_boundaries():
    bounds = [{"frame": 45, "type": "cut", "duration_frames": 0},
              {"frame": 90, "type": "dissolve", "duration_frames": 15},
              {"frame": 91, "type": "cut", "duration_frames": 0}]       # 33 ms shot: dropped
    shots = build_shots(bounds, duration_ms=5000, fps=30)
    assert shots == [
        {"start_ms": 0, "end_ms": 1500, "transition_in": {"type": "cut", "duration_ms": 0}},
        {"start_ms": 1500, "end_ms": 3000, "transition_in": {"type": "cut", "duration_ms": 0}},
        {"start_ms": 3000, "end_ms": 5000, "transition_in": {"type": "dissolve", "duration_ms": 500}},
    ]


def test_build_shots_no_boundaries():
    assert build_shots([], duration_ms=2000, fps=30) == [
        {"start_ms": 0, "end_ms": 2000, "transition_in": {"type": "cut", "duration_ms": 0}}]
```

`tests/integration/test_shots.py`:
```python
import pytest

from vtpl import ingest
from vtpl.analyzers import shots
from vtpl.util.workdir import WorkDir

pytestmark = pytest.mark.slow
FRAME_MS = 34


def _analyse(tmp_path, video):
    wd = WorkDir(tmp_path / "w")
    ingest.run(wd, video)
    shots.run(wd, video)
    return wd.read_json("shots.json")["shots"]


def test_hard_cuts(tmp_path, cards_video):
    got = [s["start_ms"] for s in _analyse(tmp_path, cards_video["path"])[1:]]
    assert len(got) == len(cards_video["cuts_ms"])
    for g, t in zip(got, cards_video["cuts_ms"]):
        assert abs(g - t) <= FRAME_MS


def test_transitions(tmp_path, transitions_video):
    got = _analyse(tmp_path, transitions_video["path"])[1:]
    truth = transitions_video["truth"]
    assert [s["transition_in"]["type"] for s in got] == [t["type"] for t in truth]
    for s, t in zip(got, truth):
        assert abs(s["start_ms"] - t["ms"]) <= 2 * FRAME_MS
        assert abs(s["transition_in"]["duration_ms"] - t["duration_ms"]) <= 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_build_shots.py tests/integration/test_shots.py -v`
Expected: FAIL with `ImportError: cannot import name 'shots'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/shots.py`

```python
from pathlib import Path

from vtpl.analyzers.framesig import luma, read_small_frames
from vtpl.analyzers.transitions import classify_boundaries
from vtpl.util.timebase import frame_to_ms
from vtpl.util.workdir import WorkDir

OUTPUT = "shots.json"
MIN_SHOT_MS = 100


def detect_cuts(video: Path, threshold: float = 3.0, min_scene_len: int = 8) -> list[int]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector

    v = open_video(str(video))
    sm = SceneManager()
    sm.add_detector(AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(v)
    return [start.get_frames() for start, _ in sm.get_scene_list()[1:]]


def build_shots(bounds: list[dict], duration_ms: int, fps: float) -> list[dict]:
    shots, start = [], 0
    tin = {"type": "cut", "duration_ms": 0}
    for b in bounds:
        ms = frame_to_ms(b["frame"], fps)
        if ms - start < MIN_SHOT_MS or duration_ms - ms < MIN_SHOT_MS:
            continue
        shots.append({"start_ms": start, "end_ms": ms, "transition_in": tin})
        start = ms
        tin = {"type": b["type"], "duration_ms": frame_to_ms(b["duration_frames"], fps)}
    shots.append({"start_ms": start, "end_ms": duration_ms, "transition_in": tin})
    return shots


def run(wd: WorkDir, source: Path) -> list[str]:
    meta = wd.read_json("meta.json")
    video = wd.path("norm.mp4")
    frames = read_small_frames(video)
    bounds = classify_boundaries(detect_cuts(video), frames, luma(frames))
    wd.write_json(OUTPUT, {"shots": build_shots(bounds, meta["duration_ms"], meta["fps"])})
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_build_shots.py tests/integration/test_shots.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/shots.py tests/unit/test_build_shots.py tests/integration/test_shots.py
git commit -m "feat: shots stage with transition types"
```

---

### Task 8: Audio stage (beats)

**Files:**
- Create: `vtpl/analyzers/audio.py`
- Test: `tests/unit/test_audio.py`

- [ ] **Step 1: Write the failing test**

```python
from vtpl.analyzers.audio import analyze_beats, beat_pattern, is_on_beat
from tests.synth.make_fixtures import click_signal


def test_click_track_beats_and_bpm():
    y, sr, truth = click_signal(seconds=8.0, bpm=120.0)
    res = analyze_beats(y, sr)
    assert abs(res["bpm"] - 120.0) <= 2.0
    matched = sum(any(abs(b - t) <= 30 for b in res["beats_ms"]) for t in truth)
    assert matched >= 0.8 * len(truth)
    assert len(res["downbeat_hint_ms"]) >= len(res["beats_ms"]) // 4


def test_silence_has_no_beats():
    import numpy as np
    res = analyze_beats(np.zeros(22050 * 3, dtype=np.float32), 22050)
    assert res == {"bpm": None, "beats_ms": [], "downbeat_hint_ms": []}


def test_is_on_beat():
    beats = [500, 1000, 1500]
    assert is_on_beat(1070, beats)
    assert not is_on_beat(1250, beats)
    assert not is_on_beat(1000, [])


def test_beat_pattern():
    shots = [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 1000, "end_ms": 1200},
             {"start_ms": 1200, "end_ms": 3200}]
    assert beat_pattern(shots, period_ms=500) == [2, 1, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.audio'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/audio.py`

```python
import bisect
from pathlib import Path

import numpy as np

from vtpl.util.workdir import WorkDir

OUTPUT = "audio.json"
EMPTY = {"bpm": None, "beats_ms": [], "downbeat_hint_ms": []}


def analyze_beats(y: np.ndarray, sr: int) -> dict:
    import librosa

    if float(np.abs(y).max(initial=0.0)) < 1e-4:
        return dict(EMPTY)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0 or len(beat_frames) == 0:
        return dict(EMPTY)
    beats_ms = [int(round(t * 1000)) for t in librosa.frames_to_time(beat_frames, sr=sr)]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_audio.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/audio.py tests/unit/test_audio.py
git commit -m "feat: audio stage with beats, BPM and beat helpers"
```

---

### Task 9: Speech stage

**Files:**
- Create: `vtpl/analyzers/speech.py`
- Test: `tests/unit/test_speech.py`

- [ ] **Step 1: Write the failing test**

```python
import os
from types import SimpleNamespace as NS

import pytest

from vtpl.analyzers import speech
from vtpl.util.workdir import WorkDir


def test_words_from_segments():
    segs = [NS(words=[NS(word=" Hello", start=0.3, end=0.52, probability=0.91),
                      NS(word=" ", start=0.52, end=0.53, probability=0.1)]),
            NS(words=None),
            NS(words=[NS(word="world!", start=0.6, end=1.0, probability=0.8)])]
    assert speech.words_from_segments(segs) == [
        {"w": "Hello", "start_ms": 300, "end_ms": 520, "p": 0.91},
        {"w": "world!", "start_ms": 600, "end_ms": 1000, "p": 0.8},
    ]


def test_no_audio_skips(tmp_path):
    wd = WorkDir(tmp_path)
    wd.write_json("meta.json", {"has_audio": False})
    assert speech.run(wd, tmp_path / "x.mp4") == ["no audio: speech skipped"]
    assert wd.read_json("speech.json") == {"language": None, "words": []}


@pytest.mark.skipif(os.environ.get("VTPL_RUN_WHISPER") != "1", reason="set VTPL_RUN_WHISPER=1 (downloads a model)")
def test_whisper_runs_on_clicks(tmp_path):
    from tests.synth.make_fixtures import click_track
    wd = WorkDir(tmp_path)
    click_track(wd.path("audio.wav"), seconds=3.0)
    wd.write_json("meta.json", {"has_audio": True})
    os.environ.setdefault("VTPL_WHISPER_MODEL", "tiny")
    assert speech.run(wd, tmp_path / "x.mp4") == []
    assert isinstance(wd.read_json("speech.json")["words"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_speech.py -v`
Expected: FAIL with `ImportError: cannot import name 'speech'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/speech.py`

```python
import os
from pathlib import Path

from vtpl.util.workdir import WorkDir

OUTPUT = "speech.json"


def words_from_segments(segments) -> list[dict]:
    out = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                out.append({"w": text, "start_ms": int(round(w.start * 1000)),
                            "end_ms": int(round(w.end * 1000)), "p": round(float(w.probability), 3)})
    return out


def run(wd: WorkDir, source: Path) -> list[str]:
    if not wd.read_json("meta.json")["has_audio"]:
        wd.write_json(OUTPUT, {"language": None, "words": []})
        return ["no audio: speech skipped"]
    from faster_whisper import WhisperModel

    model = WhisperModel(os.environ.get("VTPL_WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wd.path("audio.wav")), word_timestamps=True, vad_filter=True)
    wd.write_json(OUTPUT, {"language": info.language, "words": words_from_segments(segments)})
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_speech.py -v`
Expected: 2 passed, 1 skipped

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/speech.py tests/unit/test_speech.py
git commit -m "feat: speech stage with word timestamps"
```

---

### Task 10: Template schema, timeline checks, transition clamping

**Files:**
- Create: `vtpl/schema.py`, `vtpl/timeline.py`
- Test: `tests/unit/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
import copy

import pytest

from vtpl.schema import TemplateError, validate
from vtpl.timeline import clamp_transitions

GOOD = {
    "version": 1,
    "meta": {"w": 1080, "h": 1920, "fps": 30, "duration_ms": 3000, "bpm": 120.0, "source_id": "abc123abc123"},
    "audio": {"beats_ms": [500, 1000], "downbeat_hint_ms": [500], "beat_pattern": [2, 4], "music_slot": "music"},
    "shots": [
        {"id": "s1", "start_ms": 0, "end_ms": 1000, "slot": "clip_1", "on_beat": False,
         "transition_in": {"type": "cut", "duration_ms": 0}},
        {"id": "s2", "start_ms": 1000, "end_ms": 3000, "slot": "clip_2", "on_beat": True,
         "transition_in": {"type": "dissolve", "duration_ms": 400}},
    ],
    "texts": [],
    "captions": {"style_ref": None, "words": [{"w": "hi", "start_ms": 100, "end_ms": 300}]},
}


def test_good_template_passes():
    validate(copy.deepcopy(GOOD))


def test_schema_error_names_path():
    t = copy.deepcopy(GOOD)
    t["shots"][1]["transition_in"]["type"] = "wipe"
    with pytest.raises(TemplateError, match="shots/1/transition_in/type"):
        validate(t)


@pytest.mark.parametrize("mutate,msg", [
    (lambda t: t["shots"][0].update(start_ms=10), "shots\\[0\\].start_ms must be 0"),
    (lambda t: t["shots"][1].update(start_ms=1100), "must equal shots\\[0\\].end_ms"),
    (lambda t: t["meta"].update(duration_ms=2500), "last shot must end at meta.duration_ms"),
    (lambda t: t["shots"][1]["transition_in"].update(duration_ms=1000), "must be shorter than both"),
])
def test_timeline_errors(mutate, msg):
    t = copy.deepcopy(GOOD)
    mutate(t)
    with pytest.raises(TemplateError, match=msg):
        validate(t)


def test_clamp_transitions():
    shots = copy.deepcopy(GOOD["shots"])
    shots[1]["transition_in"]["duration_ms"] = 900
    clamp_transitions(shots)
    assert shots[1]["transition_in"] == {"type": "dissolve", "duration_ms": 500}
    shots[1]["transition_in"] = {"type": "fade", "duration_ms": 0}
    clamp_transitions(shots)
    assert shots[1]["transition_in"] == {"type": "cut", "duration_ms": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.schema'`

- [ ] **Step 3: Implement** — `vtpl/timeline.py`

```python
def clamp_transitions(shots: list[dict]) -> list[dict]:
    """Limit each transition to half the shorter neighbouring shot; zero-length becomes a cut."""
    if shots:
        shots[0]["transition_in"] = {"type": "cut", "duration_ms": 0}
    for i in range(1, len(shots)):
        tin = shots[i]["transition_in"]
        if tin["type"] == "cut":
            tin["duration_ms"] = 0
            continue
        limit = min(shots[i]["end_ms"] - shots[i]["start_ms"],
                    shots[i - 1]["end_ms"] - shots[i - 1]["start_ms"]) // 2
        tin["duration_ms"] = max(0, min(tin["duration_ms"], limit))
        if tin["duration_ms"] == 0:
            tin["type"] = "cut"
    return shots
```

`vtpl/schema.py`:
```python
from jsonschema import Draft202012Validator

INT = {"type": "integer", "minimum": 0}
MS_LIST = {"type": "array", "items": INT}

TEMPLATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["version", "meta", "audio", "shots", "texts", "captions"],
    "properties": {
        "version": {"const": 1},
        "meta": {
            "type": "object",
            "required": ["w", "h", "fps", "duration_ms", "bpm", "source_id"],
            "properties": {
                "w": {"type": "integer", "minimum": 1}, "h": {"type": "integer", "minimum": 1},
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "duration_ms": {"type": "integer", "minimum": 1},
                "bpm": {"type": ["number", "null"]},
                "source_id": {"type": "string"},
            },
        },
        "audio": {
            "type": "object",
            "required": ["beats_ms", "downbeat_hint_ms", "beat_pattern", "music_slot"],
            "properties": {
                "beats_ms": MS_LIST, "downbeat_hint_ms": MS_LIST,
                "beat_pattern": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "music_slot": {"type": "string"},
            },
        },
        "shots": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/shot"}},
        "texts": {"type": "array", "items": {"type": "object"}},
        "captions": {
            "type": "object",
            "required": ["style_ref", "words"],
            "properties": {
                "style_ref": {"type": ["string", "null"]},
                "words": {"type": "array", "items": {
                    "type": "object", "required": ["w", "start_ms", "end_ms"],
                    "properties": {"w": {"type": "string"}, "start_ms": INT, "end_ms": INT}}},
            },
        },
    },
    "$defs": {
        "shot": {
            "type": "object",
            "required": ["id", "start_ms", "end_ms", "slot", "on_beat", "transition_in"],
            "properties": {
                "id": {"type": "string"}, "start_ms": INT, "end_ms": INT, "slot": {"type": "string"},
                "on_beat": {"type": "boolean"},
                "transition_in": {
                    "type": "object", "required": ["type", "duration_ms"],
                    "properties": {"type": {"enum": ["cut", "fade", "dissolve", "flash"]}, "duration_ms": INT},
                },
            },
        },
    },
}

_VALIDATOR = Draft202012Validator(TEMPLATE_SCHEMA)


class TemplateError(ValueError):
    pass


def check_timeline(t: dict) -> None:
    shots, dur = t["shots"], t["meta"]["duration_ms"]
    if shots[0]["start_ms"] != 0:
        raise TemplateError("shots[0].start_ms must be 0")
    for i, s in enumerate(shots):
        if s["end_ms"] <= s["start_ms"]:
            raise TemplateError(f"shots[{i}] has end_ms <= start_ms")
        if i and s["start_ms"] != shots[i - 1]["end_ms"]:
            raise TemplateError(f"shots[{i}].start_ms must equal shots[{i - 1}].end_ms")
        d = s["transition_in"]["duration_ms"]
        if i and d and d >= min(s["end_ms"] - s["start_ms"], shots[i - 1]["end_ms"] - shots[i - 1]["start_ms"]):
            raise TemplateError(f"shots[{i}].transition_in.duration_ms ({d}) must be shorter than both neighbouring shots")
    if shots[-1]["end_ms"] != dur:
        raise TemplateError(f"last shot must end at meta.duration_ms ({dur})")


def validate(t: dict) -> None:
    errors = sorted(_VALIDATOR.iter_errors(t), key=lambda e: list(e.absolute_path))
    if errors:
        e = errors[0]
        loc = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise TemplateError(f"template.json invalid at {loc}: {e.message}")
    check_timeline(t)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_schema.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/schema.py vtpl/timeline.py tests/unit/test_schema.py
git commit -m "feat: template JSON schema, timeline checks, transition clamping"
```

---

### Task 11: Merge stage

**Files:**
- Create: `vtpl/merge.py`
- Test: `tests/unit/test_merge.py`

- [ ] **Step 1: Write the failing test**

```python
from vtpl import merge
from vtpl.schema import validate
from vtpl.util.workdir import WorkDir

META = {"w": 1080, "h": 1920, "fps": 30, "duration_ms": 3000, "has_audio": True}
SHOTS = {"shots": [
    {"start_ms": 0, "end_ms": 1000, "transition_in": {"type": "cut", "duration_ms": 0}},
    {"start_ms": 1000, "end_ms": 3000, "transition_in": {"type": "dissolve", "duration_ms": 1500}},
]}
AUDIO = {"bpm": 120.0, "beats_ms": [500, 1000, 1500], "downbeat_hint_ms": [500]}
SPEECH = {"language": "en", "words": [{"w": "hi", "start_ms": 100, "end_ms": 300, "p": 0.9}]}


def test_build_template():
    t = merge.build_template(META, SHOTS, AUDIO, SPEECH, source_id="abc")
    validate(t)
    assert [s["slot"] for s in t["shots"]] == ["clip_1", "clip_2"]
    assert [s["on_beat"] for s in t["shots"]] == [False, True]
    assert t["shots"][1]["transition_in"] == {"type": "dissolve", "duration_ms": 500}   # clamped
    assert t["audio"]["beat_pattern"] == [2, 4]
    assert t["captions"] == {"style_ref": None, "words": [{"w": "hi", "start_ms": 100, "end_ms": 300}]}
    assert t["meta"]["bpm"] == 120.0 and t["meta"]["source_id"] == "abc"


def test_build_template_without_audio_or_speech():
    t = merge.build_template(META, SHOTS, None, None, source_id="abc")
    validate(t)
    assert t["meta"]["bpm"] is None and t["audio"]["beat_pattern"] == []
    assert t["captions"]["words"] == []


def test_run_reads_workdir(tmp_path):
    wd = WorkDir(tmp_path / "abc123abc123")
    wd.write_json("meta.json", META)
    wd.write_json("shots.json", SHOTS)
    wd.write_json("audio.json", AUDIO)
    assert merge.run(wd, tmp_path / "v.mp4") == ["speech.json missing: no captions"]
    assert wd.read_json("template.json")["meta"]["source_id"] == "abc123abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_merge.py -v`
Expected: FAIL with `ImportError: cannot import name 'merge'`

- [ ] **Step 3: Implement** — `vtpl/merge.py`

```python
import copy
from pathlib import Path

from vtpl.analyzers.audio import beat_pattern, is_on_beat
from vtpl.schema import validate
from vtpl.timeline import clamp_transitions
from vtpl.util.workdir import WorkDir

OUTPUT = "template.json"


def build_template(meta: dict, shots_doc: dict, audio: dict | None, speech: dict | None, source_id: str) -> dict:
    audio = audio or {"bpm": None, "beats_ms": [], "downbeat_hint_ms": []}
    words = [{"w": w["w"], "start_ms": w["start_ms"], "end_ms": w["end_ms"]} for w in (speech or {}).get("words", [])]
    shots = clamp_transitions(copy.deepcopy(shots_doc["shots"]))
    out_shots = [{
        "id": f"s{i + 1}",
        "start_ms": s["start_ms"],
        "end_ms": s["end_ms"],
        "slot": f"clip_{i + 1}",
        "on_beat": i > 0 and is_on_beat(s["start_ms"], audio["beats_ms"]),
        "transition_in": s["transition_in"],
    } for i, s in enumerate(shots)]
    pattern = beat_pattern(out_shots, 60000 / audio["bpm"]) if audio["bpm"] else []
    template = {
        "version": 1,
        "meta": {"w": meta["w"], "h": meta["h"], "fps": meta["fps"], "duration_ms": meta["duration_ms"],
                 "bpm": audio["bpm"], "source_id": source_id},
        "audio": {"beats_ms": audio["beats_ms"], "downbeat_hint_ms": audio["downbeat_hint_ms"],
                  "beat_pattern": pattern, "music_slot": "music"},
        "shots": out_shots,
        "texts": [],
        "captions": {"style_ref": None, "words": words},
    }
    validate(template)
    return template


def run(wd: WorkDir, source: Path) -> list[str]:
    warnings = []
    optional = {}
    for name in ("audio.json", "speech.json"):
        if wd.has(name):
            optional[name] = wd.read_json(name)
        else:
            warnings.append(f"{name} missing: " + ("no beats" if name == "audio.json" else "no captions"))
    template = build_template(wd.read_json("meta.json"), wd.read_json("shots.json"),
                              optional.get("audio.json"), optional.get("speech.json"), source_id=wd.root.name)
    wd.write_json(OUTPUT, template)
    return warnings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_merge.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/merge.py tests/unit/test_merge.py
git commit -m "feat: merge stage builds validated template.json"
```

---

### Task 12: Pipeline runner

**Files:**
- Create: `vtpl/pipeline.py`
- Test: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from vtpl.pipeline import STAGE_NAMES, Stage, analyze


def _stages(calls, fail=None):
    def make(name, output):
        def fn(wd, source):
            calls.append(name)
            if name == fail:
                raise RuntimeError(f"{name} broke")
            wd.write_json(output, {"ok": True})
            return [f"{name} warn"] if name == "audio" else []
        return Stage(name, output, fn)
    return [make("ingest", "meta.json"), make("audio", "audio.json"), make("merge", "template.json")]


def test_default_stage_order():
    assert STAGE_NAMES == ["ingest", "shots", "audio", "speech", "merge"]


def test_runs_all_then_caches(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    calls = []
    out = analyze(src, tmp_path / "work", stages=_stages(calls))
    assert calls == ["ingest", "audio", "merge"] and out.name == "template.json"
    run = (out.parent / "run.json").read_text()
    assert "audio warn" in run
    analyze(src, tmp_path / "work", stages=_stages(calls))
    assert calls == ["ingest", "audio", "merge"]                       # all cached
    analyze(src, tmp_path / "work", force=True, only={"audio"}, stages=_stages(calls))
    assert calls[-1] == "audio" and len(calls) == 4


def test_non_critical_failure_is_recorded(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    out = analyze(src, tmp_path / "work", stages=_stages([], fail="audio"))
    import json
    run = json.loads((out.parent / "run.json").read_text())
    assert run["stages"]["audio"] == {"status": "failed", "seconds": pytest.approx(0, abs=1), "warnings": ["audio broke"]}
    assert run["stages"]["merge"]["status"] == "ok"


def test_critical_failure_raises(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    with pytest.raises(RuntimeError, match="ingest broke"):
        analyze(src, tmp_path / "work", stages=_stages([], fail="ingest"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.pipeline'`

- [ ] **Step 3: Implement** — `vtpl/pipeline.py`

```python
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vtpl import ingest, merge
from vtpl.analyzers import audio, shots, speech
from vtpl.util.workdir import WorkDir, video_id

CRITICAL = {"ingest", "shots", "merge"}


@dataclass(frozen=True)
class Stage:
    name: str
    output: str
    fn: Callable[[WorkDir, Path], list[str]]


STAGES = [
    Stage("ingest", ingest.OUTPUT, ingest.run),
    Stage("shots", shots.OUTPUT, shots.run),
    Stage("audio", audio.OUTPUT, audio.run),
    Stage("speech", speech.OUTPUT, speech.run),
    Stage("merge", merge.OUTPUT, merge.run),
]
STAGE_NAMES = [s.name for s in STAGES]


def analyze(source: Path, out_root: Path, force: bool = False, only: set[str] | None = None,
            stages: list[Stage] | None = None) -> Path:
    """Run stages in order. Returns the path of template.json."""
    wd = WorkDir(Path(out_root) / video_id(source))
    for st in stages or STAGES:
        if only and st.name not in only:
            continue
        if wd.has(st.output) and not force:
            continue
        t0 = time.perf_counter()
        try:
            warnings = st.fn(wd, source)
        except Exception as e:
            wd.record(st.name, "failed", time.perf_counter() - t0, [str(e)])
            if st.name in CRITICAL:
                raise
            continue
        wd.record(st.name, "ok", time.perf_counter() - t0, warnings)
    return wd.path("template.json")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: pipeline runner with caching and failure policy"
```

---

### Task 13: Render — media slot resolution

**Files:**
- Create: `vtpl/render/__init__.py` (empty), `vtpl/render/slots.py`
- Test: `tests/unit/test_slots.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from vtpl.render.slots import MediaError, resolve_media

T = {"shots": [{"slot": "clip_1"}, {"slot": "clip_2"}, {"slot": "clip_3"}]}


def touch(p):
    p.write_bytes(b"x")
    return p


def test_by_name_with_reuse_and_music(tmp_path):
    c1 = touch(tmp_path / "clip_1.mp4")
    c3 = touch(tmp_path / "clip_3.MOV")
    m = touch(tmp_path / "music.mp3")
    media = resolve_media(T, tmp_path)
    assert media.clips == {"clip_1": c1, "clip_2": c1, "clip_3": c3}
    assert media.music == m
    assert media.warnings == ["clip_2: no clip found, reusing clip_1.mp4"]


def test_yaml_mapping(tmp_path):
    beach = touch(tmp_path / "beach.mov")
    song = touch(tmp_path / "song.wav")
    (tmp_path / "texts.yaml").write_text(
        "clips:\n  clip_1: beach.mov\n  clip_2: beach.mov\n  clip_3: beach.mov\nmusic: song.wav\ntext_1: HELLO\n")
    media = resolve_media(T, tmp_path)
    assert set(media.clips.values()) == {beach}
    assert media.music == song and media.texts == {"text_1": "HELLO"} and media.warnings == []


def test_missing_first_clip(tmp_path):
    with pytest.raises(MediaError, match="No clip for clip_1"):
        resolve_media(T, tmp_path)


def test_mapped_file_missing(tmp_path):
    (tmp_path / "texts.yaml").write_text("clips:\n  clip_1: nope.mp4\n")
    with pytest.raises(MediaError, match="nope.mp4, which does not exist"):
        resolve_media(T, tmp_path)


def test_no_music_warns(tmp_path):
    touch(tmp_path / "clip_1.mp4")
    media = resolve_media({"shots": [{"slot": "clip_1"}]}, tmp_path)
    assert media.music is None and media.warnings == ["No music file found; output will be silent"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_slots.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.render'`

- [ ] **Step 3: Implement** — `vtpl/render/slots.py`

```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


class MediaError(ValueError):
    pass


@dataclass
class Media:
    clips: dict[str, Path]
    music: Path | None
    texts: dict[str, str]
    warnings: list[str] = field(default_factory=list)


def _existing(media_dir: Path, name: str, slot: str) -> Path:
    p = media_dir / name
    if not p.exists():
        raise MediaError(f"{slot} is mapped to {name}, which does not exist in {media_dir}")
    return p


def resolve_media(template: dict, media_dir: Path) -> Media:
    media_dir = Path(media_dir)
    cfg_path = media_dir / "texts.yaml"
    cfg = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    mapping = cfg.get("clips") or {}
    by_stem = {p.stem: p for p in sorted(media_dir.iterdir()) if p.suffix.lower() in VIDEO_EXT}

    clips, warnings, prev = {}, [], None
    for shot in template["shots"]:
        slot = shot["slot"]
        path = _existing(media_dir, mapping[slot], slot) if slot in mapping else by_stem.get(slot)
        if path is None:
            if prev is None:
                raise MediaError(f"No clip for {slot}. Add {slot}.mp4 to {media_dir} or map it under 'clips:' in texts.yaml")
            warnings.append(f"{slot}: no clip found, reusing {prev.name}")
            path = prev
        clips[slot] = prev = path

    if cfg.get("music"):
        music = _existing(media_dir, cfg["music"], "music")
    else:
        music = next((p for p in sorted(media_dir.iterdir()) if p.stem == "music" and p.suffix.lower() in AUDIO_EXT), None)
    if music is None:
        warnings.append("No music file found; output will be silent")

    texts = {k: str(v) for k, v in cfg.items() if str(k).startswith("text_")}
    return Media(clips, music, texts, warnings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_slots.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/render tests/unit/test_slots.py
git commit -m "feat: resolve user media into template slots"
```

---

### Task 14: Render — filter graph builder (pure)

**Files:**
- Create: `vtpl/render/graph.py`
- Test: `tests/unit/test_graph.py`

Background: an `xfade` overlaps two segments by its duration `d`. To keep template timing, the segment before a transition is extended by `d - d//2` and the segment after starts `d//2` earlier; the xfade `offset` is then `boundary_ms - d//2`. Hard cuts use `concat` (no overlap).

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from vtpl.render.graph import Segment, build_args, build_filter, plan_segments, trim_start_ms

T = {"shots": [
    {"slot": "clip_1", "start_ms": 0, "end_ms": 1000, "transition_in": {"type": "cut", "duration_ms": 0}},
    {"slot": "clip_2", "start_ms": 1000, "end_ms": 2500, "transition_in": {"type": "dissolve", "duration_ms": 400}},
    {"slot": "clip_3", "start_ms": 2500, "end_ms": 4000, "transition_in": {"type": "cut", "duration_ms": 0}},
], "meta": {"duration_ms": 4000}}


def test_plan_segments():
    assert plan_segments(T) == [
        Segment("clip_1", 1200, "cut", 0, 0),
        Segment("clip_2", 1700, "dissolve", 400, 800),
        Segment("clip_3", 1500, "cut", 0, 2500),
    ]


def test_trim_start_centres_clip():
    assert trim_start_ms(5000, 1000) == 2000
    assert trim_start_ms(800, 1000) == 0


def test_build_filter():
    f = build_filter(plan_segments(T), [0, 100, 200], has_music=True, duration_ms=4000)
    assert "[0:v]trim=start=0.000:duration=1.200,setpts=PTS-STARTPTS" in f
    assert "[s0][s1]xfade=transition=fade:duration=0.400:offset=0.800[j1]" in f
    assert "[j1][s2]concat=n=2:v=1:a=0[j2]" in f
    assert "[j2]trim=duration=4.000,setpts=PTS-STARTPTS[vout]" in f
    assert "[3:a]atrim=0:4.000,asetpts=PTS-STARTPTS,afade=t=out:st=3.700:d=0.3[aout]" in f


def test_build_args_without_music():
    segs = plan_segments(T)
    args = build_args(segs, [Path("a.mp4")] * 3, [0, 0, 0], None, 4000, Path("o.mp4"))
    assert args.count("-stream_loop") == 3
    assert "[aout]" not in " ".join(args)
    assert args[-1] == "o.mp4" and "+faststart" in args
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.render.graph'`

- [ ] **Step 3: Implement** — `vtpl/render/graph.py`

```python
from dataclasses import dataclass
from pathlib import Path

XFADE = {"dissolve": "fade", "fade": "fadeblack", "flash": "fadewhite"}
W, H, FPS = 1080, 1920, 30


@dataclass(frozen=True)
class Segment:
    slot: str
    length_ms: int      # shot length plus transition overlaps
    trans_in: str       # cut | fade | dissolve | flash
    trans_in_ms: int
    offset_ms: int      # where this segment starts in the output timeline (xfade offset)


def _dur(tin: dict) -> int:
    return 0 if tin["type"] == "cut" else tin["duration_ms"]


def plan_segments(template: dict) -> list[Segment]:
    shots = template["shots"]
    segs = []
    for i, s in enumerate(shots):
        d_in = _dur(s["transition_in"]) if i > 0 else 0
        d_out = _dur(shots[i + 1]["transition_in"]) if i + 1 < len(shots) else 0
        length = (s["end_ms"] - s["start_ms"]) + d_in // 2 + (d_out - d_out // 2)
        segs.append(Segment(s["slot"], length, s["transition_in"]["type"] if d_in else "cut", d_in,
                            s["start_ms"] - d_in // 2))
    return segs


def trim_start_ms(clip_ms: int, need_ms: int) -> int:
    return max(0, (clip_ms - need_ms) // 2)


def _s(ms: int) -> str:
    return f"{ms / 1000:.3f}"


def build_filter(segs: list[Segment], trims_ms: list[int], has_music: bool, duration_ms: int) -> str:
    parts = [
        f"[{i}:v]trim=start={_s(t0)}:duration={_s(seg.length_ms)},setpts=PTS-STARTPTS,"
        f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p,settb=AVTB[s{i}]"
        for i, (seg, t0) in enumerate(zip(segs, trims_ms))
    ]
    cur = "[s0]"
    for i in range(1, len(segs)):
        seg, nxt = segs[i], f"[j{i}]"
        if seg.trans_in == "cut":
            parts.append(f"{cur}[s{i}]concat=n=2:v=1:a=0{nxt}")
        else:
            parts.append(f"{cur}[s{i}]xfade=transition={XFADE[seg.trans_in]}:"
                         f"duration={_s(seg.trans_in_ms)}:offset={_s(seg.offset_ms)}{nxt}")
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
    for p in clips:
        args += ["-stream_loop", "-1", "-i", str(p)]
    if music is not None:
        args += ["-stream_loop", "-1", "-i", str(music)]
    args += ["-filter_complex", build_filter(segs, trims_ms, music is not None, duration_ms), "-map", "[vout]"]
    if music is not None:
        args += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    args += ["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-t", _s(duration_ms), str(out)]
    return args
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/render/graph.py tests/unit/test_graph.py
git commit -m "feat: pure FFmpeg filter-graph builder for cuts and transitions"
```

---

### Task 15: Render — snap to beats and render orchestration

**Files:**
- Create: `vtpl/render/snap.py`, `vtpl/render/run.py`
- Test: `tests/unit/test_snap.py`, `tests/integration/test_render.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_snap.py`:
```python
import copy

import pytest

from vtpl.render.snap import snap_template
from vtpl.schema import TemplateError, validate
from tests.unit.test_schema import GOOD


def test_snap_rescales_shots_and_words():
    t = snap_template(copy.deepcopy(GOOD), new_bpm=60.0)       # pattern [2, 4] at 1000 ms/beat
    validate(t)
    assert [(s["start_ms"], s["end_ms"]) for s in t["shots"]] == [(0, 2000), (2000, 6000)]
    assert t["meta"]["duration_ms"] == 6000 and t["meta"]["bpm"] == 60.0
    assert t["captions"]["words"][0]["start_ms"] == 200                # shot 1 stretched 1000 -> 2000 ms
    assert t["audio"]["beats_ms"] == []


def test_snap_needs_pattern():
    t = copy.deepcopy(GOOD)
    t["audio"]["beat_pattern"] = []
    with pytest.raises(TemplateError, match="no beat_pattern"):
        snap_template(t, 120.0)
```

`tests/integration/test_render.py`:
```python
import json

import pytest

from tests.synth.make_fixtures import click_track, solid_clip
from vtpl.render.run import render
from vtpl.util.ffmpeg import duration_ms, probe

pytestmark = pytest.mark.slow


def test_render_produces_video_with_audio(tmp_path):
    from tests.unit.test_schema import GOOD
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(GOOD))
    media = tmp_path / "mine"
    media.mkdir()
    solid_clip(media / "clip_1.mp4", "cyan")
    solid_clip(media / "clip_2.mp4", "magenta")
    click_track(media / "music.wav", seconds=4.0)
    out = tmp_path / "out.mp4"
    warnings = render(tpl, media, out)
    assert warnings == []
    assert abs(duration_ms(out) - 3000) <= 70
    assert {s["codec_type"] for s in probe(out)["streams"]} == {"video", "audio"}
    assert (tmp_path / "render_cmd.txt").read_text().startswith("ffmpeg ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_snap.py tests/integration/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.render.snap'`

- [ ] **Step 3: Implement** — `vtpl/render/snap.py`

```python
import copy

from vtpl.schema import TemplateError
from vtpl.timeline import clamp_transitions


def snap_template(template: dict, new_bpm: float) -> dict:
    """Re-time shots to beat_pattern at new_bpm; warp captions and texts with them."""
    t = copy.deepcopy(template)
    pattern = t["audio"]["beat_pattern"]
    if not pattern:
        raise TemplateError("template has no beat_pattern (the source had no music), so it cannot be snapped to beats")
    period = 60000.0 / new_bpm
    old = [(s["start_ms"], s["end_ms"]) for s in t["shots"]]
    new, start = [], 0
    for n in pattern:
        end = int(round(start + n * period))
        new.append((start, end))
        start = end

    def warp(ms: int) -> int:
        for (o0, o1), (n0, n1) in zip(old, new):
            if ms <= o1:
                return int(round(n0 + (ms - o0) * (n1 - n0) / (o1 - o0)))
        return new[-1][1]

    for s, (n0, n1) in zip(t["shots"], new):
        s["start_ms"], s["end_ms"] = n0, n1
    for item in [*t["captions"]["words"], *t["texts"]]:
        item["start_ms"], item["end_ms"] = warp(item["start_ms"]), warp(item["end_ms"])
    clamp_transitions(t["shots"])
    t["meta"]["duration_ms"] = start
    t["meta"]["bpm"] = round(new_bpm, 2)
    t["audio"]["beats_ms"], t["audio"]["downbeat_hint_ms"] = [], []
    return t
```

`vtpl/render/run.py`:
```python
import json
import shlex
from pathlib import Path

from vtpl.analyzers.audio import beats_of_file
from vtpl.render.graph import build_args, plan_segments, trim_start_ms
from vtpl.render.slots import MediaError, resolve_media
from vtpl.render.snap import snap_template
from vtpl.schema import validate
from vtpl.util.ffmpeg import duration_ms, require_ffmpeg, run_ffmpeg


def render(template_path: Path, media_dir: Path, out: Path, snap_to_beats: bool = False) -> list[str]:
    require_ffmpeg()
    template = json.loads(Path(template_path).read_text())
    validate(template)
    media = resolve_media(template, media_dir)
    if snap_to_beats:
        if media.music is None:
            raise MediaError("--snap-to-beats needs a music file in the media folder")
        bpm = beats_of_file(media.music)["bpm"]
        if bpm is None:
            raise MediaError(f"No beats found in {media.music.name}; render without --snap-to-beats")
        template = snap_template(template, bpm)
        validate(template)

    segs = plan_segments(template)
    clips = [media.clips[s.slot] for s in segs]
    trims = [trim_start_ms(duration_ms(c), s.length_ms) for c, s in zip(clips, segs)]
    args = build_args(segs, clips, trims, media.music, template["meta"]["duration_ms"], Path(out))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    (Path(out).parent / "render_cmd.txt").write_text("ffmpeg " + shlex.join(args) + "\n")
    run_ffmpeg(args)
    return media.warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_snap.py tests/integration/test_render.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/render/snap.py vtpl/render/run.py tests/unit/test_snap.py tests/integration/test_render.py
git commit -m "feat: render orchestration and snap-to-beats"
```

---

### Task 16: CLI

**Files:**
- Create: `vtpl/cli.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from typer.testing import CliRunner

from vtpl import cli

runner = CliRunner()


def test_help_lists_commands():
    r = runner.invoke(cli.app, ["--help"])
    assert r.exit_code == 0 and "analyze" in r.output and "render" in r.output


def test_unknown_stage_rejected(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    r = runner.invoke(cli.app, ["analyze", str(v), "--stages", "shots,bogus"])
    assert r.exit_code == 1 and "unknown stages: bogus" in r.output


def test_analyze_calls_pipeline(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "run_analyze", lambda src, out, force, only: seen.update(
        src=src, force=force, only=only) or tmp_path / "template.json")
    r = runner.invoke(cli.app, ["analyze", str(v), "--force", "--stages", "ingest,shots"])
    assert r.exit_code == 0 and "template.json" in r.output
    assert seen == {"src": v, "force": True, "only": {"ingest", "shots"}}


def test_render_reports_media_error(tmp_path, monkeypatch):
    from vtpl.render.slots import MediaError
    tpl = tmp_path / "t.json"
    tpl.write_text("{}")

    def boom(*a, **k):
        raise MediaError("No clip for clip_1")
    monkeypatch.setattr(cli, "run_render", boom)
    r = runner.invoke(cli.app, ["render", str(tpl), "--media", str(tmp_path)])
    assert r.exit_code == 1 and "No clip for clip_1" in r.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'cli'`

- [ ] **Step 3: Implement** — `vtpl/cli.py`

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/cli.py tests/unit/test_cli.py
git commit -m "feat: vtpl analyze and render CLI"
```

---

### Task 17: Round-trip tests and README

**Files:**
- Create: `tests/roundtrip/__init__.py` (empty), `tests/roundtrip/test_roundtrip.py`, `README.md`

- [ ] **Step 1: Write the round-trip test**

```python
import json

import pytest

from tests.synth.make_fixtures import solid_clip
from vtpl.pipeline import analyze
from vtpl.render.run import render

pytestmark = pytest.mark.slow
NO_SPEECH = {"ingest", "shots", "audio", "merge"}
COLORS = ["cyan", "magenta", "orange", "teal", "olive"]
FRAME_MS = 34


def _roundtrip(tmp_path, video, with_music=None):
    tpl_path = analyze(video, tmp_path / "work", only=NO_SPEECH)
    tpl = json.loads(tpl_path.read_text())
    media = tmp_path / "mine"
    media.mkdir()
    for i, shot in enumerate(tpl["shots"]):
        solid_clip(media / f"{shot['slot']}.mp4", COLORS[i % len(COLORS)])
    if with_music:
        (media / "music.wav").write_bytes(with_music.read_bytes())
    out = tmp_path / "out.mp4"
    render(tpl_path, media, out)
    again = json.loads(analyze(out, tmp_path / "work2", only=NO_SPEECH).read_text())
    return tpl, again


def test_roundtrip_hard_cuts(tmp_path, cards_video):
    tpl, again = _roundtrip(tmp_path, cards_video["path"], with_music=cards_video["wav"])
    assert [s["on_beat"] for s in tpl["shots"]] == [False, True, True, True]
    assert len(again["shots"]) == len(tpl["shots"])
    for a, b in zip(tpl["shots"], again["shots"]):
        assert abs(a["start_ms"] - b["start_ms"]) <= FRAME_MS
    assert abs(again["meta"]["bpm"] - tpl["meta"]["bpm"]) <= 2


def test_roundtrip_transitions(tmp_path, transitions_video):
    tpl, again = _roundtrip(tmp_path, transitions_video["path"])
    assert [s["transition_in"]["type"] for s in again["shots"]] == [s["transition_in"]["type"] for s in tpl["shots"]]
    for a, b in zip(tpl["shots"], again["shots"]):
        assert abs(a["start_ms"] - b["start_ms"]) <= 2 * FRAME_MS
```

- [ ] **Step 2: Run it**

Run: `pytest tests/roundtrip -v`
Expected: 2 passed. If `test_roundtrip_transitions` fails only on a flash, check that the rendered flash (FFmpeg `fadewhite`) reaches luma > 0.85; if it does not, lengthen nothing in the test — instead lower `BRIGHT` in `vtpl/analyzers/transitions.py` to the measured peak minus 0.05 and re-run Task 6 tests.

- [ ] **Step 3: Write `README.md`**

````markdown
# vtpl — video template extractor

Turns a short-form video (Reels / Shorts / TikTok) into an editable `template.json`, then renders a new video from that template with your own clips and music. Runs locally on CPU.

## Install

Needs Python 3.11+ and FFmpeg 6+ with libass.

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
```

## Use

```bash
vtpl analyze reel.mp4                 # writes work/<id>/template.json
vtpl render work/<id>/template.json --media ./mine --out out.mp4
vtpl render work/<id>/template.json --media ./mine --snap-to-beats
```

`./mine` holds `clip_1.mp4 … clip_N.mp4` and `music.mp3`. Optional `texts.yaml`:

```yaml
clips: {clip_1: beach.mov}
music: song.wav
```

Speech uses faster-whisper `small` by default; set `VTPL_WHISPER_MODEL=tiny` for speed. The first run downloads the model.

## Test

```bash
pytest                 # all tests
pytest -m "not slow"   # unit tests only, no FFmpeg
```
````

- [ ] **Step 4: Run the full suite**

Run: `pytest -v`
Expected: all tests pass (1 skipped: whisper test unless `VTPL_RUN_WHISPER=1`).

- [ ] **Step 5: Commit**

```bash
git add tests/roundtrip README.md
git commit -m "test: analyse-render-analyse round trip; add README"
```
