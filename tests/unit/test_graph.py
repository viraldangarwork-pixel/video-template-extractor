from pathlib import Path

from vtpl.render.graph import Segment, build_args, build_filter, plan_segments, trim_start_ms

T = {"shots": [
    {"slot": "clip_1", "start_ms": 0, "end_ms": 1000, "transition_in": {"type": "cut", "duration_ms": 0}},
    {"slot": "clip_2", "start_ms": 1000, "end_ms": 2500, "transition_in": {"type": "dissolve", "duration_ms": 400}},
    {"slot": "clip_3", "start_ms": 2500, "end_ms": 4000, "transition_in": {"type": "cut", "duration_ms": 0}},
], "meta": {"duration_ms": 4000}}


def test_plan_segments():
    assert plan_segments(T) == [
        Segment("clip_1", 36, "cut", 0, 0),
        Segment("clip_2", 51, "dissolve", 12, 24),
        Segment("clip_3", 45, "cut", 0, 75),
    ]


def test_plan_segments_lengths_sum_to_frame_aligned_boundaries():
    shots = [{"slot": f"clip_{i}", "start_ms": i * 1049, "end_ms": (i + 1) * 1049,
              "transition_in": {"type": "cut", "duration_ms": 0}} for i in range(6)]
    segs = plan_segments({"shots": shots, "meta": {"duration_ms": 6 * 1049}})
    assert [s.offset_frames for s in segs] == [round(i * 1049 * 30 / 1000) for i in range(6)]
    assert sum(s.length_frames for s in segs) == round(6 * 1049 * 30 / 1000)


def test_trim_start_centres_clip():
    assert trim_start_ms(5000, 1000) == 2000
    assert trim_start_ms(800, 1000) == 0


def test_build_filter():
    f = build_filter(plan_segments(T), has_music=True, duration_ms=4000)
    assert "[0:v]setpts=PTS-STARTPTS,scale=" in f and "trim=start" not in f
    assert ("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,"
            "fps=30,trim=end_frame=36,setpts=PTS-STARTPTS,format=yuv420p,settb=AVTB[s0]") in f
    assert "trim=end_frame=51," in f and "trim=end_frame=45," in f
    assert "[s0][s1]xfade=transition=fade:duration=0.400000:offset=0.800000[j1]" in f
    assert "[j1][s2]concat=n=2:v=1:a=0[j2]" in f
    assert "[j2]trim=duration=4.000,setpts=PTS-STARTPTS[vout]" in f
    assert "[3:a]atrim=0:4.000,asetpts=PTS-STARTPTS,afade=t=out:st=3.700:d=0.3[aout]" in f


def test_build_args_without_music():
    segs = plan_segments(T)
    args = build_args(segs, [Path("a.mp4")] * 3, [0, 0, 0], None, 4000, Path("o.mp4"))
    assert args.count("-stream_loop") == 3
    assert "[aout]" not in " ".join(args)
    assert args[-1] == "o.mp4" and "+faststart" in args


def test_build_args_seeks_each_clip_input():
    segs = plan_segments(T)
    args = build_args(segs, [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")], [0, 1500, 250], Path("m.mp3"), 4000,
                      Path("o.mp4"))
    assert args[:18] == ["-ss", "0.000", "-stream_loop", "-1", "-i", "a.mp4",
                         "-ss", "1.500", "-stream_loop", "-1", "-i", "b.mp4",
                         "-ss", "0.250", "-stream_loop", "-1", "-i", "c.mp4"]
    assert args[18:22] == ["-stream_loop", "-1", "-i", "m.mp3"]
