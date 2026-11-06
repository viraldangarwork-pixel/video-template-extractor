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


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_expected(name: str, doc) -> None:
    """Raise ValueError naming the file and entry if an expected-results document is malformed."""
    if not isinstance(doc, dict):
        raise ValueError(f"{name}: must be a JSON object with any of cuts_ms, bpm, texts, motion")
    if "cuts_ms" in doc and not (isinstance(doc["cuts_ms"], list) and all(_is_num(v) for v in doc["cuts_ms"])):
        raise ValueError(f"{name}: cuts_ms must be a list of times in ms")
    if "bpm" in doc and not _is_num(doc["bpm"]):
        raise ValueError(f"{name}: bpm must be a number")
    checks = {"texts": ({"text": str, "start_ms": "num", "end_ms": "num"}, "text, start_ms and end_ms"),
              "motion": ({"shot": int, "type": str}, "shot (1-based) and type")}
    for key, (fields, need) in checks.items():
        if key not in doc:
            continue
        if not isinstance(doc[key], list):
            raise ValueError(f"{name}: {key} must be a list")
        for i, e in enumerate(doc[key]):
            ok = isinstance(e, dict) and all(
                f in e and (_is_num(e[f]) if t == "num" else isinstance(e[f], t) and not isinstance(e[f], bool))
                for f, t in fields.items())
            if not ok:
                raise ValueError(f"{name}: {key}[{i}] needs {need}, got {json.dumps(e)}")


def _load_expected(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path.name}: not valid JSON ({e})") from None
    check_expected(path.name, doc)
    return doc


def failed_stages(template_path: Path) -> dict:
    """{stage: first message} for the stages run.json (next to template.json) records as failed."""
    run = Path(template_path).parent / "run.json"
    if not run.exists():
        return {}
    stages = json.loads(run.read_text()).get("stages", {})
    return {k: (v.get("warnings") or ["unknown error"])[0] for k, v in stages.items() if v.get("status") == "failed"}


def evaluate(golden: Path, work_root: Path, only: set[str] | None = None, force: bool = True) -> dict:
    """Analyse every golden video (from scratch unless force is False) and score it against its expected file.
    Every expected file is checked before any video is analysed; a missing video is skipped with a warning."""
    golden = Path(golden)
    paths = sorted(golden.glob("*.expected.json"))
    if not paths:
        raise FileNotFoundError(f"no *.expected.json files in {golden}")
    expected = {p: _load_expected(p) for p in paths}
    report, warnings = {}, []
    for exp_path, exp in expected.items():
        name = exp_path.name[: -len(".expected.json")]
        video = next((golden / f"{name}{ext}" for ext in VIDEO_EXT if (golden / f"{name}{ext}").exists()), None)
        if video is None:
            warnings.append(f"no video found for {exp_path.name} (expected {name}.mp4 or similar): skipped")
            continue
        template_path = analyze(video, work_root, force=force, only=only)
        try:
            report[name] = score(exp, json.loads(template_path.read_text()))
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError(f"{exp_path.name}: malformed entry ({type(e).__name__}: {e})") from None
        failed = failed_stages(template_path)
        if failed:
            report[name]["failed_stages"] = failed
    result = {"videos": report, "summary": summarise(report), "warnings": warnings}
    Path(work_root).mkdir(parents=True, exist_ok=True)
    (Path(work_root) / "eval_report.json").write_text(json.dumps(result, indent=2))
    return result
