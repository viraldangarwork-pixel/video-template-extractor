from pathlib import Path

import pytest

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
    assert ("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,"
            "setsar=1,trim=end_frame=36,setpts=PTS-STARTPTS,format=yuv420p,settb=AVTB[s0]") in f
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


def test_subtitles_filter_before_final_trim(tmp_path):
    from vtpl.render.graph import build_filter, plan_segments
    ass = tmp_path / "a b:c" / "out.ass"
    f = build_filter(plan_segments(T), False, 4000, ass=ass, fonts_dir=tmp_path / "fonts")
    tail = f.split(";")[-1]
    assert tail.startswith("[j2]subtitles=filename=") and "fontsdir=" in tail
    assert "a b\\\\:c" in tail                            # ':' -> '\:' (option level) -> '\\:' (graph level)
    assert tail.endswith("trim=duration=4.000,setpts=PTS-STARTPTS[vout]")
    assert "subtitles" not in build_filter(plan_segments(T), False, 4000)


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
    assert "crop=1080:1920,fps=30,scale=2160:3840,zoompan=" in f.split(";")[0]     # frame rate fixed before zoompan


def test_zoompan_filter_bounds_hand_edited_motion():
    from vtpl.render.graph import zoompan_filter
    wild = zoompan_filter({"type": "zoom_in", "scale_from": 0.5, "scale_to": 2.0, "dx": 0.9, "dy": -0.6}, 31)
    assert "z='1.27000*pow(1.50000,(on/30))'" in wild                 # scale ratio 4 -> 1.5, |dx|,|dy| -> 0.25
    assert "*0.25000*iw/zoom" in wild and "*-0.25000*ih/zoom" in wild
    out = zoompan_filter({"type": "zoom_out", "scale_from": 1.0, "scale_to": 0.1, "dx": 0.0, "dy": 0.0}, 31)
    assert "z='1.50000*pow(0.66667,(on/30))'" in out


def test_motion_rate_matches_the_shot_not_the_transition_overlap():
    zoom = {"type": "zoom_in", "scale_from": 1.0, "scale_to": 1.2, "dx": 0.04, "dy": 0.0}
    t = {"shots": [T["shots"][0], dict(T["shots"][1], motion=zoom), T["shots"][2]], "meta": T["meta"]}
    seg = plan_segments(t)[1]                     # 45 shot frames, played for 51 with the dissolve's overlap
    assert seg.length_frames == 51
    m = seg.motion
    assert m["scale_to"] / m["scale_from"] == pytest.approx(1.2 ** (51 / 45))
    assert m["dx"] == pytest.approx(0.04 * 51 / 45) and m["dy"] == 0.0
    assert plan_segments({"shots": [dict(T["shots"][0], motion=zoom)], "meta": T["meta"]})[0].motion == zoom
