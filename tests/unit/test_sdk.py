"""Unit tests for the vtpl public Python SDK (vtpl.__init__)."""
import json
import pytest
from pathlib import Path

import vtpl
from vtpl.schema import TemplateError


MINIMAL_TEMPLATE = {
    "version": 1,
    "meta": {
        "w": 1080,
        "h": 1920,
        "fps": 30,
        "duration_ms": 3000,
        "bpm": 120.0,
        "source_id": "testvideo123",
    },
    "audio": {
        "beats_ms": [500, 1000, 1500, 2000, 2500],
        "downbeat_hint_ms": [500, 2500],
        "beat_pattern": [6],
        "music_slot": "music",
    },
    "shots": [
        {
            "id": "s1",
            "start_ms": 0,
            "end_ms": 3000,
            "slot": "clip_1",
            "on_beat": False,
            "transition_in": {"type": "cut", "duration_ms": 0},
            "motion": {
                "type": "zoom_in",
                "scale_from": 1.0,
                "scale_to": 1.2,
                "dx": 0.0,
                "dy": 0.0,
            },
            "grade": {
                "luma_mean": 0.1,
                "luma_std": 0.05,
                "chroma": 0.02,
            },
        }
    ],
    "texts": [
        {
            "id": "t1",
            "slot": "text_1",
            "sample": "TEST HEADLINE",
            "kind": "overlay",
            "start_ms": 500,
            "end_ms": 2500,
            "box": {"x": 0.1, "y": 0.5, "w": 0.8, "h": 0.05},
            "style": {
                "font": "Montserrat-Bold",
                "font_confidence": 0.9,
                "size_px": 48,
                "color": "#00F0FF",
                "stroke": None,
                "shadow": False,
            },
            "anim": {
                "in": "fade",
                "in_ms": 300,
                "out": "fade",
                "out_ms": 300,
            },
        }
    ],
    "captions": {
        "style_ref": None,
        "words": [],
    },
}


def test_sdk_version_and_exports():
    assert isinstance(vtpl.__version__, str)
    for name in vtpl.__all__:
        assert hasattr(vtpl, name), f"Missing public symbol {name} in vtpl"


def test_load_and_validate_template(tmp_path: Path):
    template_file = tmp_path / "template.json"
    template_file.write_text(json.dumps(MINIMAL_TEMPLATE, indent=2), encoding="utf-8")

    loaded = vtpl.load_template(template_file)
    assert loaded["meta"]["bpm"] == 120.0
    assert loaded["shots"][0]["slot"] == "clip_1"
    assert loaded["texts"][0]["sample"] == "TEST HEADLINE"

    # Should pass without exception
    vtpl.validate_template(loaded)


def test_save_template(tmp_path: Path):
    out_file = tmp_path / "subfolder" / "saved_template.json"
    saved_path = vtpl.save_template(MINIMAL_TEMPLATE, out_file)
    assert saved_path.exists()

    content = json.loads(saved_path.read_text(encoding="utf-8"))
    assert content["meta"]["source_id"] == "testvideo123"


def test_load_template_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        vtpl.load_template(tmp_path / "non_existent.json")


def test_save_template_invalid_schema():
    invalid_template = {"version": 1, "meta": {}}
    with pytest.raises(TemplateError):
        vtpl.save_template(invalid_template, "should_fail.json")


def test_analyze_missing_video(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        vtpl.analyze(tmp_path / "does_not_exist.mp4")


def test_render_missing_template(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        vtpl.render(tmp_path / "missing_tmpl.json", tmp_path)


def test_render_invalid_template_dict(tmp_path: Path):
    with pytest.raises(TemplateError):
        vtpl.render({"invalid": "data"}, tmp_path)
