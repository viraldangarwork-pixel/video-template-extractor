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
