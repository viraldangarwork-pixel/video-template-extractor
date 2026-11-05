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
