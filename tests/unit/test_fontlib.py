import pytest
from PIL import ImageFont

from vtpl import fontlib


def test_catalog_files_match_family_names():
    cat = fontlib.catalog()
    assert len(cat) >= 30
    ids = [e.id for e in cat]
    assert len(ids) == len(set(ids)) and fontlib.FALLBACK_ID in ids
    for e in cat:
        assert ImageFont.truetype(str(fontlib.font_path(e)), 40).getname()[0] == e.family


def test_at_most_one_regular_and_one_bold_per_family():
    seen = set()
    for e in fontlib.catalog():
        assert (e.family, e.bold) not in seen
        seen.add((e.family, e.bold))


def test_entry_lookup():
    e = fontlib.entry("Montserrat-Bold")
    assert e.family == "Montserrat" and e.bold and e.weight == 700
    with pytest.raises(KeyError, match="unknown font 'Nope'"):
        fontlib.entry("Nope")


def test_ass_font_size_is_line_height():
    asc, desc = fontlib.pil_font("Montserrat-Bold", 100).getmetrics()
    assert fontlib.ass_font_size("Montserrat-Bold", 100) == asc + desc
    assert fontlib.ass_font_size("Montserrat-Bold", 100) > 100


def test_fonts_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VTPL_FONTS_DIR", str(tmp_path))
    assert fontlib.fonts_dir() == tmp_path
