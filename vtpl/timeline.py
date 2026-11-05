def clamp_transitions(shots: list[dict]) -> list[dict]:
    """Limit each transition to half the shorter neighbouring shot; zero-length becomes a cut."""
    if shots:
        shots[0]["transition_in"] = {"type": "cut", "duration_ms": 0}
    for i in range(1, len(shots)):
        tin = shots[i]["transition_in"]
        if tin["type"] == "cut":
            tin["duration_ms"] = 0
            continue
        limit = min(shots[i]["end_ms"] - shots[i]["start_ms"],
                    shots[i - 1]["end_ms"] - shots[i - 1]["start_ms"]) // 2
        tin["duration_ms"] = max(0, min(tin["duration_ms"], limit))
        if tin["duration_ms"] == 0:
            tin["type"] = "cut"
    return shots
