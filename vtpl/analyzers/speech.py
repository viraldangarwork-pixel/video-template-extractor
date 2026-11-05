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
