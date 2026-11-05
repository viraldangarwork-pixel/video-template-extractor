import copy

from vtpl.schema import TemplateError
from vtpl.timeline import clamp_transitions


def beat_phase_ms(beats_ms: list[int], period_ms: float, tol_ms: float = 40.0) -> int:
    """Phase of the beat grid (0 <= phase < period): the candidate b % period that the most beats
    agree with to within tol_ms, measured circularly so jitter around the wrap point still counts.
    Robust to a missed first beat or a stray onset, unlike taking beats_ms[0]."""
    if not beats_ms:
        return 0
    cands = [b % period_ms for b in beats_ms]

    def support(p: float) -> int:
        return sum(min((c - p) % period_ms, (p - c) % period_ms) <= tol_ms for c in cands)
    return int(round(max(cands, key=support))) % int(round(period_ms))


def snap_template(template: dict, new_bpm: float, first_beat_ms: int = 0) -> dict:
    """Re-time shots to beat_pattern at new_bpm; warp captions and texts with them.

    Cuts land on the music's beat grid (first_beat_ms + k * period). The music still starts at 0,
    so shot 0 runs from 0 to the first grid beat after its pattern length.
    """
    t = copy.deepcopy(template)
    pattern = t["audio"]["beat_pattern"]
    if not pattern:
        raise TemplateError("template has no beat_pattern (the source had no music), so it cannot be snapped to beats")
    if len(pattern) != len(t["shots"]):
        raise TemplateError(f"beat_pattern has {len(pattern)} entries but template has {len(t['shots'])} shots")
    period = 60000.0 / new_bpm
    phase = first_beat_ms % period
    first_end = phase + pattern[0] * period
    if first_end <= 0:
        first_end += period
    old = [(s["start_ms"], s["end_ms"]) for s in t["shots"]]
    new, start, beats = [], 0, 0
    for i, n in enumerate(pattern):
        beats += n if i else 0
        end = int(round(first_end + beats * period))     # from the grid, so rounding never accumulates
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
