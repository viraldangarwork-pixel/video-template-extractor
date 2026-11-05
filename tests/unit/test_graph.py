from pathlib import Path

from vtpl.render.graph import Segment, build_args, build_filter, plan_segments, trim_start_ms

T = {"shots": [
    {"slot": "clip_1", "start_ms": 0, "end_ms": 1000, "transition_in": {"type": "cut", "duration_ms": 0}},
    {"slot": "clip_2", "start_ms": 1000, "end_ms": 2500, "transition_in": {"type": "dissolve", "duration_ms": 400}},
    {"slot": "clip_3", "start_ms": 2500, "end_ms": 4000, "transition_in": {"type": "cut", "duration_ms": 0}},
], "meta": {"duration_ms": 4000}}


def test_plan_segments():
    assert plan_segments(T) == [
        Segment("clip_1", 1200, "cut", 0, 0),
        Segment("clip_2", 1700, "dissolve", 400, 800),
        Segment("clip_3", 1500, "cut", 0, 2500),
    ]


def test_trim_start_centres_clip():
    assert trim_start_ms(5000, 1000) == 2000
    assert trim_start_ms(800, 1000) == 0


def test_build_filter():
    f = build_filter(plan_segments(T), [0, 100, 200], has_music=True, duration_ms=4000)
    assert "[0:v]trim=start=0.000:duration=1.200,setpts=PTS-STARTPTS" in f
    assert "[s0][s1]xfade=transition=fade:duration=0.400:offset=0.800[j1]" in f
    assert "[j1][s2]concat=n=2:v=1:a=0[j2]" in f
    assert "[j2]trim=duration=4.000,setpts=PTS-STARTPTS[vout]" in f
    assert "[3:a]atrim=0:4.000,asetpts=PTS-STARTPTS,afade=t=out:st=3.700:d=0.3[aout]" in f


def test_build_args_without_music():
    segs = plan_segments(T)
    args = build_args(segs, [Path("a.mp4")] * 3, [0, 0, 0], None, 4000, Path("o.mp4"))
    assert args.count("-stream_loop") == 3
    assert "[aout]" not in " ".join(args)
    assert args[-1] == "o.mp4" and "+faststart" in args
