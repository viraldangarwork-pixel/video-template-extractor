from vtpl.analyzers.text.track import build_tracks, iou, is_prefix

STEP = 133


def det(text, box, score=0.9):
    return {"text": text, "box": list(box), "score": score}


def samples(rows):
    """rows: list of det lists, one per sample."""
    return [{"t_ms": i * STEP, "frame": i * 4, "dets": d} for i, d in enumerate(rows)]


A = (0.1, 0.6, 0.8, 0.05)
B = (0.3, 0.2, 0.4, 0.06)


def test_iou_and_prefix():
    assert iou(A, A) == 1.0 and iou(A, B) == 0.0
    assert is_prefix("sa", "SALE") and is_prefix("SALE", "s a") is False and not is_prefix("", "SALE")


def test_two_texts_two_tracks_sorted_by_start():
    s = samples([[det("WAIT", A)], [det("WAIT", A), det("DAY 1", B)], [det("WAIT", A), det("DAY 1", B)],
                 [det("WAIT", A), det("DAY 1", B)], [det("DAY 1", B)]])
    tr = build_tracks(s, STEP)
    assert [(t["id"], t["text"], t["start_ms"], t["end_ms"]) for t in tr] == [
        ("t1", "WAIT", 0, 3 * STEP), ("t2", "DAY 1", STEP, 4 * STEP)]
    assert [o["frame"] for o in tr[0]["obs"]] == [0, 4, 8, 12]


def test_gap_of_three_samples_bridged_four_splits():
    rows = [[det("HELLO", A)]] * 2 + [[]] * 3 + [[det("HELLO", A)]] * 2
    assert len(build_tracks(samples(rows), STEP)) == 1
    rows = [[det("HELLO", A)]] * 3 + [[]] * 4 + [[det("HELLO", A)]] * 3
    assert len(build_tracks(samples(rows), STEP)) == 2


def test_typewriter_prefix_growth_is_one_track_with_full_text():
    boxes = [(0.40, 0.4, 0.05, 0.08), (0.40, 0.4, 0.10, 0.08), (0.40, 0.4, 0.16, 0.08), (0.40, 0.4, 0.2, 0.08)]
    rows = [[det(t, b)] for t, b in zip(["S", "SA", "SAL", "SALE"], boxes)] + [[det("SALE", boxes[-1])]] * 3
    tr = build_tracks(samples(rows), STEP)
    assert len(tr) == 1 and tr[0]["text"] == "SALE" and tr[0]["box"] == list(boxes[-1])


def test_typewriter_partial_prefix_majority_still_gives_full_text():
    box = (0.40, 0.4, 0.2, 0.08)
    rows = [[det(t, box)] for t in ["S", "SA", "SAL", "SAL", "SAL", "SALE"]]
    tr = build_tracks(samples(rows), STEP)
    assert len(tr) == 1 and tr[0]["text"] == "SALE"


def test_ocr_noise_votes_and_short_blips_dropped():
    rows = [[det("WAIT FOR IT", A)], [det("WAIT F0R IT", A)], [det("WAIT FOR IT", A)], [det("X", B)]]
    tr = build_tracks(samples(rows), STEP)
    assert [t["text"] for t in tr] == ["WAIT FOR IT"]          # "X" lasts one sample (< 300 ms)
