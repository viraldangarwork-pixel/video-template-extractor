import copy

import pytest

from tests.unit.test_schema import GOOD
from vtpl import merge
from vtpl.schema import TemplateError, validate

META = {"w": 1080, "h": 1920, "fps": 30, "duration_ms": 3000, "has_audio": False}
SHOTS = {"shots": [
    {"start_ms": 0, "end_ms": 1000, "transition_in": {"type": "cut", "duration_ms": 0}},
    {"start_ms": 1000, "end_ms": 3000, "transition_in": {"type": "cut", "duration_ms": 0}},
]}
ZOOM = {"type": "zoom_in", "scale_from": 1.0, "scale_to": 1.2, "dx": 0.0, "dy": 0.0}
STATIC = {"type": "static", "scale_from": 1.0, "scale_to": 1.0, "dx": 0.0, "dy": 0.0}
GRADE = {"luma_mean": 0.52, "luma_std": 0.21, "chroma": 0.08}


def test_motion_and_grade_attached_per_shot():
    t = merge.build_template(META, SHOTS, None, None, "abc", None,
                             motion_doc={"shots": [ZOOM, STATIC]}, grade_doc={"shots": [GRADE, GRADE]})
    validate(t)
    assert t["shots"][0]["motion"] == ZOOM and t["shots"][1]["motion"] == STATIC
    assert t["shots"][1]["grade"] == GRADE


def test_mismatched_counts_are_ignored():
    t = merge.build_template(META, SHOTS, None, None, "abc", None, motion_doc={"shots": [ZOOM]})
    assert "motion" not in t["shots"][0]


def test_schema_checks_motion_and_grade():
    t = copy.deepcopy(GOOD)
    t["shots"][0]["motion"] = dict(ZOOM, type="spin")
    with pytest.raises(TemplateError, match="shots/0/motion/type"):
        validate(t)
    t["shots"][0]["motion"] = ZOOM
    t["shots"][0]["grade"] = {"luma_mean": 0.5}
    with pytest.raises(TemplateError, match="shots/0/grade"):
        validate(t)


def test_run_reads_motion_and_grade(tmp_path):
    from vtpl.util.workdir import WorkDir
    wd = WorkDir(tmp_path / "abc")
    wd.write_json("meta.json", META)
    wd.write_json("shots.json", SHOTS)
    wd.write_json("motion.json", {"shots": [ZOOM, STATIC]})
    wd.write_json("grade.json", {"shots": [GRADE, GRADE]})
    warnings = merge.run(wd, tmp_path / "v.mp4")
    assert not any("motion" in w or "grade" in w for w in warnings)
    assert wd.read_json("template.json")["shots"][0]["motion"]["type"] == "zoom_in"


@pytest.mark.parametrize("field,value", [("scale_to", 3.0), ("scale_from", 0.2), ("dx", 0.8), ("dy", -0.7)])
def test_schema_bounds_motion(field, value):
    t = copy.deepcopy(GOOD)
    t["shots"][0]["motion"] = dict(ZOOM, **{field: value})
    with pytest.raises(TemplateError, match=f"shots/0/motion/{field}"):
        validate(t)
