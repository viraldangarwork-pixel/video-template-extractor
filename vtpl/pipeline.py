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
    dirty = False  # an upstream stage ran, so template.json would be stale
    for st in stages or STAGES:
        if st.name == "merge" and dirty:
            pass
        elif only and st.name not in only:
            continue
        elif wd.has(st.output) and not force:
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
        dirty = True
    return wd.path("template.json")
