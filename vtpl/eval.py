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


def evaluate(golden: Path, work_root: Path, only: set[str] | None = None, force: bool = True) -> dict:
    """Analyse every golden video (from scratch unless force is False) and score it against its expected file."""
    golden = Path(golden)
    report = {}
    for exp_path in sorted(golden.glob("*.expected.json")):
        name = exp_path.name[: -len(".expected.json")]
        video = next((golden / f"{name}{ext}" for ext in VIDEO_EXT if (golden / f"{name}{ext}").exists()), None)
        if video is None:
            raise FileNotFoundError(f"no video found for {exp_path.name} (expected {name}.mp4 or similar)")
        template = json.loads(analyze(video, work_root, force=force, only=only).read_text())
        report[name] = score(json.loads(exp_path.read_text()), template)
    if not report:
        raise FileNotFoundError(f"no *.expected.json files in {golden}")
    result = {"videos": report, "summary": summarise(report)}
    Path(work_root).mkdir(parents=True, exist_ok=True)
    (Path(work_root) / "eval_report.json").write_text(json.dumps(result, indent=2))
    return result
