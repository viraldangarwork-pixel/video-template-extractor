"""Per-shot colour grade as target statistics: luma mean, luma spread and chroma (all 0..1)."""
from pathlib import Path

import cv2
import numpy as np

from vtpl import frames as fr
from vtpl.analyzers.motion import shot_window
from vtpl.util.workdir import WorkDir

OUTPUT = "grade.json"
SAMPLES = 5


def frame_stats(bgr: np.ndarray) -> tuple[float, float, float]:
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255
    y = ycc[..., 0]
    chroma = np.sqrt((ycc[..., 1] - 0.5) ** 2 + (ycc[..., 2] - 0.5) ** 2)
    return float(y.mean()), float(y.std()), float(chroma.mean())


def stats(frames: list[np.ndarray]) -> dict:
    vals = np.array([frame_stats(f) for f in frames])
    med = np.median(vals, axis=0)
    return {"luma_mean": round(float(med[0]), 4), "luma_std": round(float(med[1]), 4),
            "chroma": round(float(med[2]), 4)}


def sample_indices(first: int, last: int, n: int = SAMPLES) -> list[int]:
    if last <= first:
        return [first]
    return sorted({round(first + (last - first) * k / (n - 1)) for k in range(n)})


def run(wd: WorkDir, source: Path) -> list[str]:
    if not wd.has(fr.OUTPUT):
        raise FileNotFoundError(f"{fr.OUTPUT} missing: run the frames stage first")
    cache = wd.read_json(fr.OUTPUT)
    shots = wd.read_json("shots.json")["shots"]
    n = len(cache["files"])
    out = []
    for i in range(len(shots)):
        first, last = shot_window(shots, i, cache["fps"])
        first, last = max(0, min(first, n - 1)), max(0, min(last, n - 1))
        if last < first:
            first = last = max(0, min(round(shots[i]["start_ms"] * cache["fps"] / 1000), n - 1))
        out.append(stats([fr.cached_frame(wd, k) for k in sample_indices(first, last)]))
    wd.write_json(OUTPUT, {"shots": out})
    return []
