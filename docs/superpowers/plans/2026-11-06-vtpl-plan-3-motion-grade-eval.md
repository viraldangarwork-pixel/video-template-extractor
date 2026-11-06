# vtpl Plan 3 — Camera Motion, Colour Grade and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vtpl analyze` records each shot's camera motion (zoom in/out, pan, static) and colour grade; `vtpl render` reproduces both on the user's clips; `vtpl eval` scores the analyser against a folder of hand-checked videos.

**Architecture:** Two new optional stages, `motion` and `grade`, read the 15 fps frame cache per shot and write per-shot JSON that `merge` attaches to `template.shots[i]`. The renderer adds an `eq` filter (grade) and a `zoompan` filter (motion) to each segment's chain. `vtpl/eval.py` runs the pipeline on golden videos and compares the templates with expected JSON.

**Tech Stack:** Plan 1–2 stack; OpenCV optical flow (`goodFeaturesToTrack`, `calcOpticalFlowPyrLK`, `estimateAffinePartial2D`); FFmpeg `eq` and `zoompan`.

**Spec:** `docs/superpowers/specs/2026-11-05-video-template-extractor-design.md` §6.6, §6.7, §7.2, §9.5.

**Design change from the spec (approved by the plan author):** the spec stores grade as eq parameters relative to a fixed "neutral" reference. That cannot reproduce a look: the same eq applied to a dark clip and a bright clip gives different results. Instead the template stores the shot's **target statistics** (`luma_mean`, `luma_std`, `chroma`), and the renderer measures each user clip and computes the `eq` that maps the clip's statistics to the target (clamped to the spec's ranges: brightness −0.3..0.3, contrast 0.7..1.5, saturation 0.5..1.8).

**Branch:** `viral/feature/vtpl-motion-grade-eval`, created from `main` after Plan 2 is merged.

---

## File structure (Plan 3)

| File | Responsibility |
|---|---|
| `vtpl/analyzers/motion.py` | Per-shot zoom/pan from optical flow on cached frames |
| `vtpl/analyzers/grade.py` | Per-shot luma mean/std and chroma from cached frames |
| `vtpl/schema.py`, `vtpl/merge.py`, `vtpl/pipeline.py` | Optional `motion`/`grade` on shots; stage registration |
| `vtpl/render/grade.py` | Measure a user clip; compute `eq` parameters for a target grade |
| `vtpl/render/graph.py`, `vtpl/render/run.py` | `eq` + `zoompan` in segment chains |
| `vtpl/eval.py`, `vtpl/cli.py` | Golden-set scoring and the `vtpl eval` command |
| `tests/synth/make_fixtures.py` | `texture_clip`, `zoom_clip`, `graded_clip` fixtures |

---

### Task 1: Template fields for motion and grade

**Files:**
- Modify: `vtpl/schema.py`, `vtpl/merge.py`
- Test: `tests/unit/test_merge_motion_grade.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_merge_motion_grade.py`

```python
import copy

import pytest

from tests.unit.test_schema import GOOD
from vtpl import merge
from vtpl.schema import TemplateError, validate

META = {"w": 1080, "h": 1920, "fps": 30, "duration_ms": 3000, "has_audio": False}
SHOTS = {"shots": [
    {"start_ms": 0, "end_ms": 1000, "transition_in": {"type": "cut", "duration_ms": 0}},
    {"start_ms": 1000, "end_ms": 3000, "transition_in": {"type": "cut", "duration_ms": 0}},
]}
ZOOM = {"type": "zoom_in", "scale_from": 1.0, "scale_to": 1.2, "dx": 0.0, "dy": 0.0}
STATIC = {"type": "static", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.0, "dy": 0.0}
GRADE = {"luma_mean": 0.52, "luma_std": 0.21, "chroma": 0.08}


def test_motion_and_grade_attached_per_shot():
    t = merge.build_template(META, SHOTS, None, None, "abc", None,
                             motion_doc={"shots": [ZOOM, STATIC]}, grade_doc={"shots": [GRADE, GRADE]})
    validate(t)
    assert t["shots"][0]["motion"] == ZOOM and t["shots"][1]["motion"] == STATIC
    assert t["shots"][1]["grade"] == GRADE


def test_mismatched_counts_are_ignored():
    t = merge.build_template(META, SHOTS, None, None, "abc", None, motion_doc={"shots": [ZOOM]})
    assert "motion" not in t["shots"][0]


def test_schema_checks_motion_and_grade():
    t = copy.deepcopy(GOOD)
    t["shots"][0]["motion"] = dict(ZOOM, type="spin")
    with pytest.raises(TemplateError, match="shots/0/motion/type"):
        validate(t)
    t["shots"][0]["motion"] = ZOOM
    t["shots"][0]["grade"] = {"luma_mean": 0.5}
    with pytest.raises(TemplateError, match="shots/0/grade"):
        validate(t)


def test_run_reads_motion_and_grade(tmp_path):
    from vtpl.util.workdir import WorkDir
    wd = WorkDir(tmp_path / "abc")
    wd.write_json("meta.json", META)
    wd.write_json("shots.json", SHOTS)
    wd.write_json("motion.json", {"shots": [ZOOM, STATIC]})
    wd.write_json("grade.json", {"shots": [GRADE, GRADE]})
    warnings = merge.run(wd, tmp_path / "v.mp4")
    assert not any("motion" in w or "grade" in w for w in warnings)
    assert wd.read_json("template.json")["shots"][0]["motion"]["type"] == "zoom_in"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_merge_motion_grade.py -v`
Expected: FAIL with `TypeError: build_template() got an unexpected keyword argument 'motion_doc'`

- [ ] **Step 3: Implement**

In `vtpl/schema.py`, inside `$defs.shot.properties` add:
```python
                "motion": {
                    "type": "object", "required": ["type", "scale_from", "scale_to", "dx", "dy"],
                    "properties": {"type": {"enum": ["static", "zoom_in", "zoom_out", "pan"]},
                                   "scale_from": {"type": "number", "exclusiveMinimum": 0},
                                   "scale_to": {"type": "number", "exclusiveMinimum": 0},
                                   "dx": {"type": "number"}, "dy": {"type": "number"}},
                },
                "grade": {
                    "type": "object", "required": ["luma_mean", "luma_std", "chroma"],
                    "properties": {"luma_mean": UNIT, "luma_std": UNIT, "chroma": UNIT},
                },
```
(`UNIT` already exists from Plan 2; if it is defined below `TEMPLATE_SCHEMA`, move its definition above.)

In `vtpl/merge.py`:
```python
def _per_shot(doc: dict | None, n: int) -> list | None:
    """The doc's per-shot list when it matches the shot count, else None."""
    items = (doc or {}).get("shots")
    return items if isinstance(items, list) and len(items) == n else None
```
Extend `build_template`'s signature with `motion_doc: dict | None = None, grade_doc: dict | None = None` (after `text_doc`), and after `out_shots` is built:
```python
    for key, doc in (("motion", motion_doc), ("grade", grade_doc)):
        per = _per_shot(doc, len(out_shots))
        if per:
            for shot, value in zip(out_shots, per):
                shot[key] = value
```
In `run`, add `"motion.json"` and `"grade.json"` to the optional inputs with warnings `"no camera motion"` / `"no colour grade"` when missing, and pass `motion_doc=optional.get("motion.json"), grade_doc=optional.get("grade.json")`.

- [ ] **Step 4: Run tests**

Run: `/home/user/.venv/bin/pytest tests/unit -q`
Expected: all pass. `tests/unit/test_merge.py::test_run_reads_workdir` asserts the warnings list — append `"motion.json missing: no camera motion"` and `"grade.json missing: no colour grade"` to its expected list in the order `run` emits them.

- [ ] **Step 5: Commit**

```bash
git add vtpl/schema.py vtpl/merge.py tests/unit/test_merge_motion_grade.py tests/unit/test_merge.py
git commit -m "feat: optional camera motion and colour grade on template shots"
```

---

### Task 2: Camera motion analyser

**Files:**
- Create: `vtpl/analyzers/motion.py`
- Test: `tests/unit/test_motion.py`

Background: track corner features between consecutive cached frames, fit a similarity transform (scale + rotation + translation) per pair with RANSAC, and accumulate scale (product) and the movement of the frame centre (sum, normalised by frame size). Because the first/last frames and transitions are skipped, the measured change is extrapolated to the whole shot length.

- [ ] **Step 1: Write the failing test**

```python
import cv2
import numpy as np
import pytest

from vtpl.analyzers.motion import STATIC, classify, shot_window, track_motion

H, W = 960, 540


def texture(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (H, W), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    for _ in range(60):
        x, y = rng.integers(0, W - 40), rng.integers(0, H - 40)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 30, int(y) + 30), int(rng.integers(0, 255)), -1)
    return img


def warp(img, scale=1.0, tx=0.0, ty=0.0):
    m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REFLECT)


def test_zoom_in_sequence():
    base = texture()
    frames = [warp(base, 1.02 ** i) for i in range(10)]
    scale, dx, dy = track_motion(frames)
    assert scale == pytest.approx(1.02 ** 9, abs=0.02) and abs(dx) < 0.01 and abs(dy) < 0.01


def test_pan_sequence():
    base = texture()
    frames = [warp(base, tx=4 * i) for i in range(10)]
    scale, dx, dy = track_motion(frames)
    assert scale == pytest.approx(1.0, abs=0.01) and dx == pytest.approx(36 / W, abs=0.01) and abs(dy) < 0.01


def test_flat_frames_have_no_motion_estimate():
    assert track_motion([np.zeros((H, W), np.uint8)] * 5) is None


def test_classify():
    assert classify(1.1, 0, 0)["type"] == "zoom_in"
    assert classify(0.9, 0, 0)["type"] == "zoom_out"
    assert classify(1.0, 0.05, 0) == {"type": "pan", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.05, "dy": 0.0}
    assert classify(1.01, 0.01, 0.0) == STATIC


def test_shot_window_skips_transitions_and_edges():
    shots = [{"start_ms": 0, "end_ms": 2000, "transition_in": {"type": "cut", "duration_ms": 0}},
             {"start_ms": 2000, "end_ms": 4000, "transition_in": {"type": "dissolve", "duration_ms": 400}}]
    assert shot_window(shots, 0, 15) == (1, 25)        # ends 200 ms before the dissolve's centre, minus 1 frame
    assert shot_window(shots, 1, 15) == (34, 58)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_motion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.motion'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/motion.py`

```python
"""Per-shot camera motion (zoom / pan / static) from optical flow on cached frames."""
import math
from pathlib import Path

import cv2
import numpy as np

from vtpl import frames as fr
from vtpl.util.workdir import WorkDir

OUTPUT = "motion.json"
MIN_POINTS = 20
RESEED = 50
ZOOM_MIN = 0.04            # |scale - 1| for a zoom
PAN_MIN = 0.03             # centre movement, share of the frame, for a pan
EDGE_FRAMES = 1            # cached frames skipped at each end of a shot
STATIC = {"type": "static", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.0, "dy": 0.0}


def _features(gray: np.ndarray):
    return cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=8)


def track_motion(frames_gray: list[np.ndarray]) -> tuple[float, float, float] | None:
    """(total scale, centre dx, centre dy as shares of the frame), or None if too little can be tracked."""
    h, w = frames_gray[0].shape
    centre = np.array([w / 2, h / 2])
    scale, dx, dy, ok = 1.0, 0.0, 0.0, 0
    pts = _features(frames_gray[0])
    for a, b in zip(frames_gray, frames_gray[1:]):
        if pts is None or len(pts) < RESEED:
            pts = _features(a)
        if pts is None or len(pts) < MIN_POINTS:
            pts = None
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
        keep = status.reshape(-1) == 1
        good_a, good_b = pts[keep], nxt[keep]
        if len(good_a) < MIN_POINTS:
            pts = None
            continue
        m, _ = cv2.estimateAffinePartial2D(good_a, good_b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if m is None:
            pts = None
            continue
        moved = m[:, :2] @ centre + m[:, 2]
        scale *= math.hypot(m[0, 0], m[1, 0])
        dx += (moved[0] - centre[0]) / w
        dy += (moved[1] - centre[1]) / h
        ok += 1
        pts = good_b.reshape(-1, 1, 2)
    if ok < max(1, (len(frames_gray) - 1) // 2):
        return None
    return scale, dx, dy


def classify(scale: float, dx: float, dy: float) -> dict:
    if scale >= 1 + ZOOM_MIN:
        kind = "zoom_in"
    elif scale <= 1 / (1 + ZOOM_MIN):
        kind = "zoom_out"
    elif math.hypot(dx, dy) >= PAN_MIN:
        kind = "pan"
    else:
        return dict(STATIC)
    return {"type": kind, "scale_from": 1.0, "scale_to": round(scale, 3) if kind != "pan" else 1.0,
            "dx": round(dx, 3), "dy": round(dy, 3)}


def _half(tin: dict) -> int:
    return 0 if tin["type"] == "cut" else tin["duration_ms"] // 2


def shot_window(shots: list[dict], i: int, cache_fps: int) -> tuple[int, int]:
    """First and last cached frame index to analyse for shot i (inclusive)."""
    s = shots[i]
    lo = s["start_ms"] + (_half(s["transition_in"]) if i else 0)
    hi = s["end_ms"] - (_half(shots[i + 1]["transition_in"]) if i + 1 < len(shots) else 0)
    first = math.ceil(lo * cache_fps / 1000) + EDGE_FRAMES
    last = math.floor(hi * cache_fps / 1000) - 1 - EDGE_FRAMES
    return first, last


def run(wd: WorkDir, source: Path) -> list[str]:
    if not wd.has(fr.OUTPUT):
        raise FileNotFoundError(f"{fr.OUTPUT} missing: run the frames stage first")
    cache = wd.read_json(fr.OUTPUT)
    shots = wd.read_json("shots.json")["shots"]
    out, warnings = [], []
    for i, s in enumerate(shots):
        first, last = shot_window(shots, i, cache["fps"])
        last = min(last, len(cache["files"]) - 1)
        if last - first < 2:
            out.append(dict(STATIC))
            continue
        gray = [cv2.cvtColor(fr.cached_frame(wd, k), cv2.COLOR_BGR2GRAY) for k in range(first, last + 1)]
        res = track_motion(gray)
        if res is None:
            warnings.append(f"shot {i + 1}: too few trackable points, marked static")
            out.append(dict(STATIC))
            continue
        scale, dx, dy = res
        k = (s["end_ms"] - s["start_ms"]) / ((last - first) * 1000 / cache["fps"])   # extrapolate to the shot
        out.append(classify(scale ** k, dx * k, dy * k))
    wd.write_json(OUTPUT, {"shots": out})
    return warnings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_motion.py -v`
Expected: 5 passed. (`shot_window` arithmetic: shot 0 ends at 2000 − 200 = 1800 ms → floor(27) − 1 − 1 = 25; shot 1 starts at 2200 ms → ceil(33) + 1 = 34 and ends at 4000 ms → 60 − 1 − 1 = 58.)

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/motion.py tests/unit/test_motion.py
git commit -m "feat: per-shot camera motion from optical flow"
```

---

### Task 3: Colour grade analyser

**Files:**
- Create: `vtpl/analyzers/grade.py`
- Test: `tests/unit/test_grade.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from vtpl.analyzers.grade import frame_stats, sample_indices, stats


def test_gray_frame():
    y, sd, c = frame_stats(np.full((20, 20, 3), 128, np.uint8))
    assert y == pytest.approx(128 / 255, abs=0.01) and sd == pytest.approx(0, abs=1e-6) and c < 0.01


def test_contrast_and_chroma_increase():
    flat = np.full((20, 20, 3), 128, np.uint8)
    stripes = flat.copy()
    stripes[::2] = 30
    stripes[1::2] = 230
    red = np.zeros((20, 20, 3), np.uint8)
    red[..., 2] = 220
    assert frame_stats(stripes)[1] > 0.3 > frame_stats(flat)[1]
    assert frame_stats(red)[2] > 0.15


def test_stats_is_median_over_frames():
    frames = [np.full((10, 10, 3), v, np.uint8) for v in (50, 128, 200)]
    s = stats(frames)
    assert s["luma_mean"] == pytest.approx(128 / 255, abs=0.01) and set(s) == {"luma_mean", "luma_std", "chroma"}


def test_sample_indices():
    assert sample_indices(10, 30, 5) == [10, 15, 20, 25, 30]
    assert sample_indices(4, 5, 5) == [4, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/user/.venv/bin/pytest tests/unit/test_grade.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vtpl.analyzers.grade'`

- [ ] **Step 3: Implement** — `vtpl/analyzers/grade.py`

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/user/.venv/bin/pytest tests/unit/test_grade.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add vtpl/analyzers/grade.py tests/unit/test_grade.py
git commit -m "feat: per-shot colour grade statistics"
```

---

### Task 4: Register stages; fixtures; pipeline integration

**Files:**
- Modify: `vtpl/pipeline.py`, `tests/unit/test_pipeline.py`, `tests/synth/make_fixtures.py`
- Test: `tests/integration/test_motion_grade_stages.py`

- [ ] **Step 1: Add fixtures to `tests/synth/make_fixtures.py`**

```python
def texture_image(path: Path, seed: int = 0) -> None:
    """A 1080x1920 PNG with lots of trackable detail (noise plus coloured blocks)."""
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (1920, 1080, 3), dtype=np.uint8), (0, 0), 3)
    for _ in range(300):
        x, y = int(rng.integers(0, 1000)), int(rng.integers(0, 1840))
        cv2.rectangle(img, (x, y), (x + 60, y + 60), [int(v) for v in rng.integers(0, 255, 3)], -1)
    cv2.imwrite(str(path), img)


def texture_clip(path: Path, image: Path, seconds: float = 3.0, vf: str | None = None) -> None:
    """A still clip of `image`, optionally passed through a video filter."""
    args = ["-loop", "1", "-framerate", str(FPS), "-i", str(image), "-t", str(seconds)]
    if vf:
        args += ["-vf", vf]
    _ff([*args, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(path)])


def zoom_clip(path: Path, image: Path, seconds: float = 3.0, scale_to: float = 1.25) -> float:
    """A clip zooming into `image` from 1.0 to scale_to over its length; returns scale_to."""
    n = round(seconds * FPS)
    vf = (f"scale=2160:3840,zoompan=z='pow({scale_to},on/{n - 1})':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'"
          f":d=1:s=1080x1920:fps={FPS}")
    texture_clip(path, image, seconds, vf)
    return scale_to
```

- [ ] **Step 2: Write the failing tests**

`tests/integration/test_motion_grade_stages.py`:
```python
import pytest

from tests.synth.make_fixtures import texture_clip, texture_image, zoom_clip
from vtpl.pipeline import analyze

pytestmark = pytest.mark.slow
STAGES = {"ingest", "frames", "shots", "motion", "grade", "merge"}


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    p = tmp_path_factory.mktemp("tex") / "tex.png"
    texture_image(p)
    return p


def _template(tmp_path, video):
    import json
    return json.loads(analyze(video, tmp_path / "work", only=STAGES).read_text())


def test_zoom_detected(tmp_path, image):
    video = tmp_path / "zoom.mp4"
    zoom_clip(video, image, 3.0, 1.25)
    t = _template(tmp_path, video)
    assert len(t["shots"]) == 1
    m = t["shots"][0]["motion"]
    assert m["type"] == "zoom_in" and m["scale_to"] == pytest.approx(1.25, abs=0.06)


def test_still_is_static_and_graded(tmp_path, image):
    video = tmp_path / "still.mp4"
    texture_clip(video, image, 3.0)
    t = _template(tmp_path, video)
    assert t["shots"][0]["motion"]["type"] == "static"
    g = t["shots"][0]["grade"]
    assert 0.2 < g["luma_mean"] < 0.8 and g["luma_std"] > 0.05


def test_eq_brightening_shows_in_grade(tmp_path, image):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    texture_clip(a, image, 2.0)
    texture_clip(b, image, 2.0, "eq=brightness=0.15:saturation=1.6")
    ga = _template(tmp_path / "A", a)["shots"][0]["grade"]
    gb = _template(tmp_path / "B", b)["shots"][0]["grade"]
    assert gb["luma_mean"] > ga["luma_mean"] + 0.08 and gb["chroma"] > ga["chroma"] * 1.3
```

In `tests/unit/test_pipeline.py`, update `test_default_stage_order`:
```python
def test_default_stage_order():
    assert STAGE_NAMES == ["ingest", "frames", "shots", "motion", "grade", "audio", "speech", "text", "merge"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/home/user/.venv/bin/pytest tests/integration/test_motion_grade_stages.py tests/unit/test_pipeline.py -v`
Expected: FAIL — stage order assertion; `KeyError: 'motion'` in the integration tests.

- [ ] **Step 4: Implement** — in `vtpl/pipeline.py` import `from vtpl.analyzers import audio, grade, motion, shots, speech` and insert after the shots stage:
```python
    Stage("motion", motion.OUTPUT, motion.run),
    Stage("grade", grade.OUTPUT, grade.run),
```
(`CRITICAL` is unchanged — both are optional.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/user/.venv/bin/pytest tests/integration/test_motion_grade_stages.py tests/unit/test_pipeline.py -v`
Expected: all pass. If the zoom scale is off, print the per-pair scales from `track_motion` and check `shot_window`/the extrapolation factor before touching thresholds.

- [ ] **Step 6: Commit**

```bash
git add vtpl/pipeline.py tests/unit/test_pipeline.py tests/synth/make_fixtures.py tests/integration/test_motion_grade_stages.py
git commit -m "feat: register motion and grade stages"
```

---

### Task 5: Render camera motion with zoompan

**Files:**
- Modify: `vtpl/render/graph.py`
- Test: `tests/unit/test_graph.py` (append), `tests/integration/test_render_motion.py`

Background: zoompan crops a window from a 2× upscaled frame (2× avoids jitter from whole-pixel steps). Zoom follows `z0·S^p` with p = 0..1 over the segment, which matches the per-frame multiplicative zoom the analyser measures. A pan moves the window opposite to the content: for content moving `dx` of the frame width, the window's x moves by −dx·iw/zoom; a base zoom of 1 + |dx| (+2 %) leaves room for it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_graph.py`:
```python
def test_zoompan_filter_expressions():
    from vtpl.render.graph import zoompan_filter
    z = zoompan_filter({"type": "zoom_in", "scale_from": 1.0, "scale_to": 1.2, "dx": 0.0, "dy": 0.0}, 91)
    assert z.startswith("scale=2160:3840,zoompan=z='1.00000*pow(1.20000,(on/90))'")
    assert ":d=1:s=1080x1920:fps=30" in z
    p = zoompan_filter({"type": "pan", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.1, "dy": 0.0}, 31)
    assert "z='1.12000*pow(1.00000,(on/30))'" in p and "-((on/30)-0.5)*0.10000*iw/zoom" in p


def test_segments_carry_motion_and_static_is_dropped():
    from vtpl.render.graph import build_filter, plan_segments
    t = {"shots": [dict(T["shots"][0], motion={"type": "zoom_in", "scale_from": 1.0, "scale_to": 1.2, "dx": 0, "dy": 0}),
                   dict(T["shots"][1], motion={"type": "static", "scale_from": 1.0, "scale_to": 1.0, "dx": 0, "dy": 0}),
                   T["shots"][2]], "meta": T["meta"]}
    segs = plan_segments(t)
    assert segs[0].motion["type"] == "zoom_in" and segs[1].motion is None and segs[2].motion is None
    f = build_filter(segs, False, 4000)
    assert f.split(";")[0].count("zoompan") == 1 and "zoompan" not in f.split(";")[1]
```

`tests/integration/test_render_motion.py`:
```python
import copy
import json

import pytest

from tests.synth.make_fixtures import texture_clip, texture_image
from tests.unit.test_schema import GOOD
from vtpl.pipeline import analyze
from vtpl.render.run import render

pytestmark = pytest.mark.slow
STAGES = {"ingest", "frames", "shots", "motion", "merge"}


def _one_shot(motion):
    t = copy.deepcopy(GOOD)
    t["shots"] = [{"id": "s1", "start_ms": 0, "end_ms": 3000, "slot": "clip_1", "on_beat": False,
                   "transition_in": {"type": "cut", "duration_ms": 0}, "motion": motion}]
    t["meta"]["duration_ms"] = 3000
    t["audio"]["beat_pattern"] = []
    t["captions"]["words"] = []          # no burned-in captions over the measured picture
    return t


@pytest.mark.parametrize("motion,check", [
    ({"type": "zoom_in", "scale_from": 1.0, "scale_to": 1.25, "dx": 0.0, "dy": 0.0},
     lambda m: m["type"] == "zoom_in" and abs(m["scale_to"] - 1.25) <= 0.06),
    ({"type": "pan", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.1, "dy": 0.0},
     lambda m: m["type"] == "pan" and abs(m["dx"] - 0.1) <= 0.03 and abs(m["dy"]) <= 0.02),
])
def test_rendered_motion_is_measured_back(tmp_path, motion, check):
    image = tmp_path / "tex.png"
    texture_image(image)
    media = tmp_path / "mine"
    media.mkdir()
    texture_clip(media / "clip_1.mp4", image, 4.0)
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps(_one_shot(motion)))
    out = tmp_path / "out.mp4"
    render(tpl, media, out)
    again = json.loads(analyze(out, tmp_path / "work", only=STAGES).read_text())
    assert check(again["shots"][0]["motion"]), again["shots"][0]["motion"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/user/.venv/bin/pytest tests/unit/test_graph.py tests/integration/test_render_motion.py -v`
Expected: FAIL — `cannot import name 'zoompan_filter'`; `Segment` has no `motion`.

- [ ] **Step 3: Implement** — in `vtpl/render/graph.py`

Add fields with defaults to `Segment` (after `offset_frames`):
```python
    motion: dict | None = None     # non-static camera motion to reproduce
    grade: dict | None = None      # target grade statistics (Task 6)
```
In `plan_segments`, pass `motion=m if m and m["type"] != "static" else None` and `grade=s.get("grade")` where `m = s.get("motion")`.

Add:
```python
def zoompan_filter(m: dict, n_frames: int) -> str:
    s = max(m["scale_to"] / m["scale_from"], 1e-3)
    dx, dy = float(m["dx"]), float(m["dy"])
    shift = max(abs(dx), abs(dy))
    zb = 1 + shift + 0.02 if shift > 0 else 1.0
    z0 = zb * max(1.0, 1 / s)
    p = f"(on/{max(1, n_frames - 1)})"
    x = f"max(0,min(iw-iw/zoom,(iw-iw/zoom)/2-({p}-0.5)*{dx:.5f}*iw/zoom))"
    y = f"max(0,min(ih-ih/zoom,(ih-ih/zoom)/2-({p}-0.5)*{dy:.5f}*ih/zoom))"
    return (f"scale={2 * W}:{2 * H},zoompan=z='{z0:.5f}*pow({s:.5f},{p})':x='{x}':y='{y}'"
            f":d=1:s={W}x{H}:fps={FPS}")
```
In `build_filter`, build each segment chain with optional parts inserted before `setsar=1` (so a chain with neither stays byte-identical to today's):
```python
    parts = []
    for i, seg in enumerate(segs):
        extra = ""
        if eqs and eqs[i]:
            extra += eqs[i] + ","
        if seg.motion:
            extra += zoompan_filter(seg.motion, seg.length_frames) + ","
        parts.append(f"[{i}:v]setpts=PTS-STARTPTS,"
                     f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{extra}setsar=1,fps={FPS},"
                     f"trim=end_frame={seg.length_frames},setpts=PTS-STARTPTS,format=yuv420p,settb=AVTB[s{i}]")
```
and add the keyword parameter `eqs: list[str | None] | None = None` to `build_filter` (after `duration_ms`, before `ass`) and to `build_args`. **Update the existing call inside `build_args` to pass everything after `duration_ms` by keyword** (`build_filter(segs, music is not None, duration_ms, eqs=eqs, ass=ass, fonts_dir=fonts_dir)`) — today it passes `ass, fonts_dir` positionally, which would now land in `eqs`. Task 6 fills `eqs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/user/.venv/bin/pytest tests/unit/test_graph.py tests/integration/test_render_motion.py tests/integration/test_render.py -v`
Expected: all pass. If the pan comes back with the wrong sign or size, render one frame at p=0 and p=1 and measure the content shift directly before changing the formula.

- [ ] **Step 5: Commit**

```bash
git add vtpl/render/graph.py tests/unit/test_graph.py tests/integration/test_render_motion.py
git commit -m "feat: reproduce camera zoom and pan with zoompan"
```

---

### Task 6: Render colour grade with eq

**Files:**
- Create: `vtpl/render/grade.py`
- Modify: `vtpl/render/run.py`
- Test: `tests/unit/test_render_grade.py`, `tests/integration/test_render_grade.py`

Background: FFmpeg `eq` maps luma as `(Y − 0.5)·contrast + 0.5 + brightness` and scales chroma by `saturation`. Given the clip's statistics (m, s, c) and the target's (M, S, C): contrast = S/s, brightness = M − ((m − 0.5)·contrast + 0.5), saturation = C/c, each clamped to the spec's range.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_render_grade.py`:
```python
import pytest

from vtpl.render.grade import eq_filter, eq_params

CLIP = {"luma_mean": 0.4, "luma_std": 0.2, "chroma": 0.05}


def test_eq_params_maps_clip_to_target():
    p = eq_params({"luma_mean": 0.55, "luma_std": 0.25, "chroma": 0.08}, CLIP)
    assert p["contrast"] == pytest.approx(1.25) and p["saturation"] == pytest.approx(1.6)
    assert p["brightness"] == pytest.approx(0.55 - ((0.4 - 0.5) * 1.25 + 0.5))


def test_eq_params_clamped_and_neutral():
    p = eq_params({"luma_mean": 1.0, "luma_std": 0.9, "chroma": 0.5}, CLIP)
    assert p == {"brightness": 0.3, "contrast": 1.5, "saturation": 1.8}
    assert eq_params(CLIP, CLIP) is None
    assert eq_params(CLIP, {"luma_mean": 0.4, "luma_std": 0.0, "chroma": 0.0})["contrast"] == 1.0


def test_eq_filter():
    assert eq_filter({"brightness": 0.1, "contrast": 1.2, "saturation": 0.9}) == \
        "eq=brightness=0.100:contrast=1.200:saturation=0.900"
```

`tests/integration/test_render_grade.py`:
```python
import copy
import json

import pytest

from tests.synth.make_fixtures import texture_clip, texture_image
from tests.unit.test_schema import GOOD
from vtpl.pipeline import analyze
from vtpl.render.run import render

pytestmark = pytest.mark.slow
STAGES = {"ingest", "frames", "shots", "grade", "merge"}


def test_grade_round_trip(tmp_path):
    image = tmp_path / "tex.png"
    texture_image(image)
    graded = tmp_path / "graded.mp4"
    texture_clip(graded, image, 3.0, "eq=brightness=0.12:contrast=1.3:saturation=1.6")
    target = json.loads(analyze(graded, tmp_path / "w1", only=STAGES).read_text())["shots"][0]["grade"]

    t = copy.deepcopy(GOOD)
    t["shots"] = [{"id": "s1", "start_ms": 0, "end_ms": 3000, "slot": "clip_1", "on_beat": False,
                   "transition_in": {"type": "cut", "duration_ms": 0}, "grade": target}]
    t["meta"]["duration_ms"] = 3000
    t["audio"]["beat_pattern"] = []
    t["captions"]["words"] = []
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps(t))
    media = tmp_path / "mine"
    media.mkdir()
    texture_clip(media / "clip_1.mp4", image, 4.0)             # the same picture, ungraded
    out = tmp_path / "out.mp4"
    render(tpl, media, out)
    got = json.loads(analyze(out, tmp_path / "w2", only=STAGES).read_text())["shots"][0]["grade"]
    for k in ("luma_mean", "luma_std", "chroma"):
        assert got[k] == pytest.approx(target[k], abs=0.03), (k, got, target)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/user/.venv/bin/pytest tests/unit/test_render_grade.py tests/integration/test_render_grade.py -v`
Expected: FAIL — `No module named 'vtpl.render.grade'`.

- [ ] **Step 3: Implement**

`vtpl/render/grade.py`:
```python
"""Match a user clip's look to a template shot's grade with FFmpeg eq."""
from pathlib import Path

import cv2

from vtpl.analyzers.grade import SAMPLES, stats

LIMITS = {"brightness": (-0.3, 0.3), "contrast": (0.7, 1.5), "saturation": (0.5, 1.8)}


def _clamp(name: str, v: float) -> float:
    lo, hi = LIMITS[name]
    return round(min(hi, max(lo, v)), 3)


def clip_stats(path: Path, start_ms: int, dur_ms: int, n: int = SAMPLES) -> dict:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for k in range(n):
            cap.set(cv2.CAP_PROP_POS_MSEC, start_ms + dur_ms * k / max(1, n - 1))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"could not read frames from {path} to match its colour grade")
    return stats(frames)


def eq_params(target: dict, clip: dict) -> dict | None:
    """eq settings that move `clip` towards `target`, or None if no change is needed."""
    c = _clamp("contrast", target["luma_std"] / clip["luma_std"]) if clip["luma_std"] > 1e-3 else 1.0
    b = _clamp("brightness", target["luma_mean"] - ((clip["luma_mean"] - 0.5) * c + 0.5))
    s = _clamp("saturation", target["chroma"] / clip["chroma"]) if clip["chroma"] > 1e-3 else 1.0
    if abs(b) < 0.01 and abs(c - 1) < 0.02 and abs(s - 1) < 0.02:
        return None
    return {"brightness": b, "contrast": c, "saturation": s}


def eq_filter(p: dict) -> str:
    return f"eq=brightness={p['brightness']:.3f}:contrast={p['contrast']:.3f}:saturation={p['saturation']:.3f}"
```

In `vtpl/render/run.py`, after `trims` is computed:
```python
    from vtpl.render.grade import clip_stats, eq_filter, eq_params
    eqs = []
    for seg, clip, t0 in zip(segs, clips, trims):
        p = eq_params(seg.grade, clip_stats(clip, t0, frames_to_ms(seg.length_frames))) if seg.grade else None
        eqs.append(eq_filter(p) if p else None)
```
(move the import to the top of the file) and pass `eqs=eqs` to `build_args`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/user/.venv/bin/pytest tests/unit/test_render_grade.py tests/integration/test_render_grade.py tests/integration/test_render.py -v`
Expected: all pass. If luma_mean is consistently off by a few hundredths, the cause is eq working on limited-range YUV (brightness is scaled by 219/255); correct `b` by that factor in `eq_params` and add a unit test, rather than widening the tolerance.

- [ ] **Step 5: Commit**

```bash
git add vtpl/render/grade.py vtpl/render/run.py tests/unit/test_render_grade.py tests/integration/test_render_grade.py
git commit -m "feat: match each clip to the template's colour grade"
```

---

### Task 7: `vtpl eval` golden-set scoring

**Files:**
- Create: `vtpl/eval.py`
- Modify: `vtpl/cli.py`
- Test: `tests/unit/test_eval.py`, `tests/integration/test_eval_cli.py`

Golden folder format: for each `name.mp4` (or `.mov`/`.mkv`/`.webm`/`.m4v`) a `name.expected.json` with any of
`{"cuts_ms": [...], "bpm": 120, "texts": [{"text": "...", "start_ms": 0, "end_ms": 0}], "motion": [{"shot": 1, "type": "zoom_in"}]}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_eval.py`:
```python
import pytest

from vtpl.eval import format_report, match_times, score, summarise


def test_match_times():
    assert match_times([1000, 2000, 3000], [1010, 2100, 3020], 34) == (2, 1, 1)
    assert match_times([], [], 34) == (0, 0, 0)


TEMPLATE = {"meta": {"bpm": 119.5},
            "shots": [{"start_ms": 0, "motion": {"type": "static"}}, {"start_ms": 1500, "motion": {"type": "zoom_in"}}],
            "texts": [{"kind": "overlay", "sample": "WAIT FOR IT", "start_ms": 520, "end_ms": 2990},
                      {"kind": "caption", "sample": "hello", "start_ms": 0, "end_ms": 100}]}


def test_score_each_field():
    s = score({"cuts_ms": [1500, 2500], "bpm": 120, "texts": [{"text": "WAIT FOR IT", "start_ms": 500, "end_ms": 3000}],
               "motion": [{"shot": 2, "type": "zoom_in"}, {"shot": 1, "type": "pan"}]}, TEMPLATE)
    assert s["cuts"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3)}
    assert s["bpm"] == {"expected": 120, "got": 119.5, "ok": True}
    assert s["texts"] == {"found": 1.0, "timing_ok": 1.0}
    assert s["motion"] == {"accuracy": 0.5}


def test_score_skips_missing_fields_and_summary():
    assert score({}, TEMPLATE) == {}
    report = {"a": {"cuts": {"f1": 1.0}, "bpm": {"ok": True}}, "b": {"cuts": {"f1": 0.5}, "bpm": {"ok": False}}}
    summary = summarise(report)
    assert summary == {"cuts_f1": 0.75, "bpm_ok": 0.5}
    assert "cuts_f1" in format_report(report, summary)
```

`tests/integration/test_eval_cli.py`:
```python
import json
import shutil

import pytest
from typer.testing import CliRunner

from vtpl import cli

pytestmark = pytest.mark.slow


def test_eval_on_cards(tmp_path, cards_video):
    golden = tmp_path / "golden"
    golden.mkdir()
    shutil.copy(cards_video["path"], golden / "cards.mp4")
    (golden / "cards.expected.json").write_text(json.dumps({"cuts_ms": cards_video["cuts_ms"], "bpm": 120}))
    r = CliRunner().invoke(cli.app, ["eval", "--golden", str(golden), "--out", str(tmp_path / "work"),
                                     "--stages", "ingest,shots,audio,merge"])
    assert r.exit_code == 0, r.output
    assert "cuts_f1" in r.output
    report = json.loads((tmp_path / "work" / "eval_report.json").read_text())
    assert report["videos"]["cards"]["cuts"]["f1"] == 1.0 and report["videos"]["cards"]["bpm"]["ok"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/user/.venv/bin/pytest tests/unit/test_eval.py tests/integration/test_eval_cli.py -v`
Expected: FAIL — `No module named 'vtpl.eval'`.

- [ ] **Step 3: Implement**

`vtpl/eval.py`:
```python
"""Score analysis against hand-checked expected results (a "golden" folder)."""
import json
from pathlib import Path

from rapidfuzz import fuzz

from vtpl.pipeline import analyze

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v")
CUT_TOL_MS, TEXT_TOL_MS, BPM_TOL = 34, 100, 2.0


def match_times(expected: list[int], got: list[int], tol: int) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) with one-to-one nearest matching."""
    remaining = sorted(got)
    tp = 0
    for e in sorted(expected):
        best = min(remaining, key=lambda g: abs(g - e), default=None)
        if best is not None and abs(best - e) <= tol:
            remaining.remove(best)
            tp += 1
    return tp, len(remaining), len(expected) - tp


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def score(expected: dict, template: dict) -> dict:
    out = {}
    if "cuts_ms" in expected:
        got = [s["start_ms"] for s in template["shots"][1:]]
        out["cuts"] = _prf(*match_times(expected["cuts_ms"], got, CUT_TOL_MS))
    if "bpm" in expected:
        got = template["meta"]["bpm"]
        out["bpm"] = {"expected": expected["bpm"], "got": got,
                      "ok": got is not None and abs(got - expected["bpm"]) <= BPM_TOL}
    if "texts" in expected and expected["texts"]:
        overlays = [t for t in template["texts"] if t["kind"] == "overlay"]
        found = timing = 0
        for e in expected["texts"]:
            best = max(overlays, key=lambda t: fuzz.ratio(t["sample"].upper(), e["text"].upper()), default=None)
            if best is not None and fuzz.ratio(best["sample"].upper(), e["text"].upper()) >= 80:
                found += 1
                if abs(best["start_ms"] - e["start_ms"]) <= TEXT_TOL_MS and abs(best["end_ms"] - e["end_ms"]) <= TEXT_TOL_MS:
                    timing += 1
        n = len(expected["texts"])
        out["texts"] = {"found": found / n, "timing_ok": timing / n}
    if "motion" in expected and expected["motion"]:
        hits = sum(1 for e in expected["motion"]
                   if 1 <= e["shot"] <= len(template["shots"])
                   and template["shots"][e["shot"] - 1].get("motion", {}).get("type") == e["type"])
        out["motion"] = {"accuracy": hits / len(expected["motion"])}
    return out


def summarise(report: dict) -> dict:
    metrics = {"cuts_f1": ("cuts", "f1"), "bpm_ok": ("bpm", "ok"), "texts_found": ("texts", "found"),
               "texts_timing_ok": ("texts", "timing_ok"), "motion_accuracy": ("motion", "accuracy")}
    out = {}
    for name, (section, key) in metrics.items():
        vals = [float(v[section][key]) for v in report.values() if section in v]
        if vals:
            out[name] = round(sum(vals) / len(vals), 3)
    return out


def format_report(report: dict, summary: dict) -> str:
    lines = []
    for name, s in sorted(report.items()):
        parts = [f"{k}={json.dumps(v)}" for k, v in s.items()]
        lines.append(f"{name}: " + "  ".join(parts))
    lines.append("summary: " + "  ".join(f"{k}={v}" for k, v in summary.items()))
    return "\n".join(lines)


def evaluate(golden: Path, work_root: Path, only: set[str] | None = None) -> dict:
    golden = Path(golden)
    report = {}
    for exp_path in sorted(golden.glob("*.expected.json")):
        name = exp_path.name[: -len(".expected.json")]
        video = next((golden / f"{name}{ext}" for ext in VIDEO_EXT if (golden / f"{name}{ext}").exists()), None)
        if video is None:
            raise FileNotFoundError(f"no video found for {exp_path.name} (expected {name}.mp4 or similar)")
        template = json.loads(analyze(video, work_root, only=only).read_text())
        report[name] = score(json.loads(exp_path.read_text()), template)
    if not report:
        raise FileNotFoundError(f"no *.expected.json files in {golden}")
    result = {"videos": report, "summary": summarise(report)}
    Path(work_root).mkdir(parents=True, exist_ok=True)
    (Path(work_root) / "eval_report.json").write_text(json.dumps(result, indent=2))
    return result
```

In `vtpl/cli.py` add:
```python
@app.command("eval")
def eval_cmd(
    golden: Path = typer.Option(..., exists=True, file_okay=False, help="Folder of videos + <name>.expected.json"),
    out: Path = typer.Option(Path("work"), help="Work folder root (eval_report.json is written here)"),
    stages: str = typer.Option("", help=f"Comma-separated subset of: {','.join(STAGE_NAMES)}"),
):
    """Score analysis against hand-checked expected results."""
    from vtpl.eval import evaluate, format_report
    only = {s.strip() for s in stages.split(",") if s.strip()} or None
    unknown = sorted((only or set()) - set(STAGE_NAMES))
    if unknown:
        _fail(f"unknown stages: {', '.join(unknown)}. Valid stages: {', '.join(STAGE_NAMES)}")
    try:
        require_ffmpeg()
        result = evaluate(golden, out, only)
    except (FFmpegError, TemplateError, ValueError, OSError) as e:
        _fail(str(e))
    typer.echo(format_report(result["videos"], result["summary"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/user/.venv/bin/pytest tests/unit/test_eval.py tests/integration/test_eval_cli.py tests/unit/test_cli.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add vtpl/eval.py vtpl/cli.py tests/unit/test_eval.py tests/integration/test_eval_cli.py
git commit -m "feat: vtpl eval scores analysis against a golden set"
```

---

### Task 8: README and full suite

**Files:** Modify `README.md`

- [ ] **Step 1: Add to README "Use"**

````markdown
Each shot also gets `motion` (zoom in/out, pan or static) and `grade` (target brightness, contrast and
colour). `vtpl render` reproduces the camera move and matches each of your clips to the shot's look.

Check accuracy on your own hand-checked videos:

```bash
vtpl eval --golden ./golden          # golden/<name>.mp4 + golden/<name>.expected.json
```

`<name>.expected.json` may contain `cuts_ms`, `bpm`, `texts` (`text`, `start_ms`, `end_ms`) and
`motion` (`shot` (1-based), `type`). Results are printed and saved to `work/eval_report.json`.
````

- [ ] **Step 2: Run the full suite**

Run: `/home/user/.venv/bin/pytest -q`
Expected: all pass (1 skipped: the Whisper test).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: motion, grade and vtpl eval"
```
