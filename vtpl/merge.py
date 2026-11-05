import copy
from pathlib import Path

from vtpl.analyzers.audio import beat_pattern, is_on_beat
from vtpl.schema import validate
from vtpl.timeline import clamp_transitions
from vtpl.util.workdir import WorkDir

OUTPUT = "template.json"


def build_template(meta: dict, shots_doc: dict, audio: dict | None, speech: dict | None, source_id: str) -> dict:
    dur = meta["duration_ms"]
    audio = audio or {"bpm": None, "beats_ms": [], "downbeat_hint_ms": []}
    audio = {**audio, **{k: [ms for ms in audio[k] if 0 <= ms <= dur] for k in ("beats_ms", "downbeat_hint_ms")}}
    words = [{"w": w["w"], "start_ms": w["start_ms"], "end_ms": w["end_ms"]} for w in (speech or {}).get("words", [])
             if 0 <= w["start_ms"] and w["end_ms"] <= dur]
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
