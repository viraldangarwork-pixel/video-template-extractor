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


def test_ass_font_size_uses_win_metrics_like_libass():
    # libass (like GDI) sizes a font so that usWinAscent + usWinDescent equals the ASS Fontsize;
    # Montserrat's hhea line height (968 + 251) is much smaller than its win one (1109 + 453).
    assert fontlib.ass_font_size("Montserrat-Bold", 100) == 156
    assert fontlib.ass_font_size("BebasNeue-Regular", 100) == 130     # win 950 + 350
    assert fontlib.ass_font_size("Anton-Regular", 96) == round(96 * (2876 + 674) / 2048)


def test_ass_font_size_matches_os2_table_for_every_bundled_font():
    from fontTools.ttLib import TTFont
    for e in fontlib.catalog():
        t = TTFont(fontlib.font_path(e))
        os2, upm = t["OS/2"], t["head"].unitsPerEm
        assert fontlib.ass_font_size(e.id, 120) == round(120 * (os2.usWinAscent + os2.usWinDescent) / upm), e.id


def test_fonts_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VTPL_FONTS_DIR", str(tmp_path))
    assert fontlib.fonts_dir() == tmp_path
