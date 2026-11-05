import copy

from vtpl.schema import TemplateError
from vtpl.timeline import clamp_transitions


def snap_template(template: dict, new_bpm: float) -> dict:
    """Re-time shots to beat_pattern at new_bpm; warp captions and texts with them."""
    t = copy.deepcopy(template)
    pattern = t["audio"]["beat_pattern"]
    if not pattern:
        raise TemplateError("template has no beat_pattern (the source had no music), so it cannot be snapped to beats")
    if len(pattern) != len(t["shots"]):
        raise TemplateError(f"beat_pattern has {len(pattern)} entries but template has {len(t['shots'])} shots")
    period = 60000.0 / new_bpm
    old = [(s["start_ms"], s["end_ms"]) for s in t["shots"]]
    new, start = [], 0
    for n in pattern:
        end = int(round(start + n * period))
        new.append((start, end))
        start = end

    def warp(ms: int) -> int:
        for (o0, o1), (n0, n1) in zip(old, new):
            if ms <= o1:
                return int(round(n0 + (ms - o0) * (n1 - n0) / (o1 - o0)))
        return new[-1][1]

    for s, (n0, n1) in zip(t["shots"], new):
        s["start_ms"], s["end_ms"] = n0, n1
    for item in [*t["captions"]["words"], *t["texts"]]:
        item["start_ms"], item["end_ms"] = warp(item["start_ms"]), warp(item["end_ms"])
    clamp_transitions(t["shots"])
    t["meta"]["duration_ms"] = start
    t["meta"]["bpm"] = round(new_bpm, 2)
    t["audio"]["beats_ms"], t["audio"]["downbeat_hint_ms"] = [], []
    return t
