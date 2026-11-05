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
