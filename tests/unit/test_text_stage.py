from vtpl.analyzers.text import stage


def _track(texts):
    return {"text": texts[-1], "obs": [{"text": t} for t in texts]}


def test_start_search_window_reaches_back_past_unreadable_typewriter_prefix():
    # OCR never reads a lone "S": the first sample of a typewriter "SALE" already shows "SAL",
    # and the reveal began up to an animation length before that.
    assert stage.start_window_ms(_track(["SAL", "SALE", "SALE"])) == stage.WINDOW_MS + stage.ANIM_MS
    assert stage.start_window_ms(_track(["sa", "SALE"])) == stage.WINDOW_MS + stage.ANIM_MS


def test_start_search_window_is_normal_when_first_sample_is_the_full_text():
    assert stage.start_window_ms(_track(["SALE", "SALE"])) == stage.WINDOW_MS
    assert stage.start_window_ms(_track(["WAIT F0R IT", "WAIT FOR IT", "WAIT FOR IT"])) == stage.WINDOW_MS
