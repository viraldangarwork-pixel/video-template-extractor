"""Download the bundled font set from Fontsource (Google Fonts; OFL or Apache-2.0) and write catalog.json.

Run once from the repo root:  python scripts/fetch_fonts.py
Then commit vtpl/assets/fonts/. The family name in catalog.json is read from each file, so it always
matches what libass and Pillow see.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont  # dev-time only (pip install fonttools); not a runtime dependency
from PIL import ImageFont

DEST = Path(__file__).resolve().parents[1] / "vtpl" / "assets" / "fonts"
URL = "https://cdn.jsdelivr.net/fontsource/fonts/{slug}@latest/latin-{weight}-normal.ttf"

FONTS = [  # (id, fontsource slug, weight) — at most one regular (<600) and one bold (>=600) per family
    ("Montserrat-Regular", "montserrat", 400), ("Montserrat-Bold", "montserrat", 700),
    ("Poppins-Regular", "poppins", 400), ("Poppins-Bold", "poppins", 700),
    ("Roboto-Regular", "roboto", 400), ("Roboto-Bold", "roboto", 700),
    ("BebasNeue-Regular", "bebas-neue", 400), ("Anton-Regular", "anton", 400),
    ("Oswald-Bold", "oswald", 700), ("OpenSans-Bold", "open-sans", 700),
    ("Lato-Bold", "lato", 700), ("Raleway-Bold", "raleway", 700),
    ("LeagueSpartan-Bold", "league-spartan", 700), ("DMSans-Bold", "dm-sans", 700),
    ("Nunito-Bold", "nunito", 700), ("Rubik-Bold", "rubik", 700),
    ("WorkSans-Bold", "work-sans", 700), ("Inter-Bold", "inter", 700),
    ("ArchivoBlack-Regular", "archivo-black", 400), ("Bangers-Regular", "bangers", 400),
    ("LuckiestGuy-Regular", "luckiest-guy", 400), ("PermanentMarker-Regular", "permanent-marker", 400),
    ("Pacifico-Regular", "pacifico", 400), ("Lobster-Regular", "lobster", 400),
    ("DancingScript-Bold", "dancing-script", 700), ("Caveat-Bold", "caveat", 700),
    ("PlayfairDisplay-Bold", "playfair-display", 700), ("Merriweather-Bold", "merriweather", 700),
    ("Lora-Bold", "lora", 700), ("AbrilFatface-Regular", "abril-fatface", 400),
    ("Righteous-Regular", "righteous", 400), ("Fredoka-Bold", "fredoka", 700),
    ("Kanit-Bold", "kanit", 700), ("BarlowCondensed-Bold", "barlow-condensed", 700),
    ("Teko-Bold", "teko", 700), ("AlfaSlabOne-Regular", "alfa-slab-one", 400),
    ("BlackOpsOne-Regular", "black-ops-one", 400), ("Bungee-Regular", "bungee", 400),
    ("Staatliches-Regular", "staatliches", 400), ("TitanOne-Regular", "titan-one", 400),
    ("PassionOne-Regular", "passion-one", 400), ("FjallaOne-Regular", "fjalla-one", 400),
]


def normalize_names(path: Path, weight: int) -> None:
    """Fontsource's static instances of variable fonts keep the default instance's names
    (e.g. 'Montserrat Thin' for the Bold file). Rewrite them to '<Family>' + Regular/Bold so
    libass/fontconfig and Pillow agree and the family is the plain name."""
    font = TTFont(path, recalcTimestamp=False)     # byte-identical on re-runs
    names = font["name"]
    family = names.getBestFamilyName() or ""
    prev = None
    while prev != family:
        prev, family = family, re.sub(r"\s+(Thin|ExtraLight|Light|\d+pt)$", "", family)
    style = "Bold" if weight >= 600 else "Regular"
    names.names = [n for n in names.names if n.nameID not in (1, 2, 3, 4, 6, 16, 17, 21, 22, 25)]
    for plat, enc, lang in ((3, 1, 0x409), (1, 0, 0)):
        names.setName(family, 1, plat, enc, lang)
        names.setName(style, 2, plat, enc, lang)
        names.setName(f"{family} {style}", 4, plat, enc, lang)
        names.setName(f"{family.replace(' ', '')}-{style}", 6, plat, enc, lang)
    os2 = font["OS/2"]
    os2.usWeightClass = weight
    os2.fsSelection = (os2.fsSelection & ~0x61) | (0x20 if weight >= 600 else 0x40)
    font["head"].macStyle = 1 if weight >= 600 else 0
    font.save(path)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    catalog, failed = [], []
    for font_id, slug, weight in FONTS:
        file = f"{slug}-{weight}.ttf"
        path = DEST / file
        try:
            if not path.exists():
                urllib.request.urlretrieve(URL.format(slug=slug, weight=weight), path)
            normalize_names(path, weight)                     # idempotent; also fixes older downloads
            family = ImageFont.truetype(str(path), 40).getname()[0]
        except Exception as e:
            failed.append(f"{font_id}: {e}")
            path.unlink(missing_ok=True)
            continue
        catalog.append({"id": font_id, "family": family, "weight": weight, "file": file, "slug": slug})
    (DEST / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"{len(catalog)} fonts written to {DEST}")
    if failed:
        print("FAILED (remove these from FONTS and re-run):\n" + "\n".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
