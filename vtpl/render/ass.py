"""Build an ASS subtitle document for text overlays and karaoke captions."""
from vtpl.fontlib import FALLBACK_ID, ass_font_size, entry

PLAY_W, PLAY_H = 1080, 1920
DEFAULT_CAPTION_STYLE = {"font": FALLBACK_ID, "size_px": 64, "color": "#FFFFFF",
                         "stroke": {"color": "#000000", "px": 4}, "shadow": False}
DEFAULT_CAPTION_BOX = {"x": 0.1, "y": 0.72, "w": 0.8, "h": 0.06}
LINE_WORDS, LINE_GAP_MS = 3, 600

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""
EVENTS = """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(ms: int) -> str:
    cs = max(0, round(ms / 10))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def style_line(name: str, st: dict, secondary_alpha: int = 0) -> str:
    e = entry(st["font"])
    stroke = st.get("stroke")
    outline, outline_col = (stroke["px"], ass_color(stroke["color"])) if stroke else (0, "&H00000000")
    fields = [name, e.family, ass_font_size(st["font"], st["size_px"]), ass_color(st["color"]),
              ass_color(st["color"], secondary_alpha), outline_col, "&H64000000", -1 if e.bold else 0,
              0, 0, 0, 100, 100, 0, 0, 1, outline, 3 if st["shadow"] else 0, 5, 0, 0, 0, 1]
    return "Style: " + ",".join(str(f) for f in fields)


def _centre(box: dict) -> tuple[int, int]:
    return round((box["x"] + box["w"] / 2) * PLAY_W), round((box["y"] + box["h"] / 2) * PLAY_H)


def _dialogue(start: int, end: int, style: str, tags: str, body: str) -> str:
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{{{tags}}}{body}"


def overlay_events(item: dict, text: str) -> list[str]:
    if text == "":
        return []
    start, end = item["start_ms"], item["end_ms"]
    dur = end - start
    a = item["anim"]
    in_ms, out_ms = min(a["in_ms"], dur // 2), min(a["out_ms"], dur // 2)
    cx, cy = _centre(item["box"])
    pos = f"\\pos({cx},{cy})"
    if a["in"] == "slide":
        off = a.get("in_from", {"dx": -0.3, "dy": 0.0})
        pos = f"\\move({cx + round(off['dx'] * PLAY_W)},{cy + round(off['dy'] * PLAY_H)},{cx},{cy},0,{in_ms})"
    elif a["out"] == "slide":
        off = a.get("out_to", {"dx": 0.3, "dy": 0.0})
        pos = f"\\move({cx},{cy},{cx + round(off['dx'] * PLAY_W)},{cy + round(off['dy'] * PLAY_H)},{dur - out_ms},{dur})"
    in_tags, out_tags = "", ""
    if a["in"] == "pop":
        k = round(in_ms * 2 / 3)
        in_tags = f"\\fscx80\\fscy80\\t(0,{k},\\fscx115\\fscy115)\\t({k},{in_ms},\\fscx100\\fscy100)"
    if a["out"] == "pop":
        out_tags = f"\\t({dur - out_ms},{dur},\\fscx0\\fscy0)"
    fade_in = in_ms if a["in"] == "fade" else 0
    fade_out = out_ms if a["out"] == "fade" else 0
    fad = f"\\fad({fade_in},{fade_out})" if fade_in or fade_out else ""
    name = item["id"]

    if a["in"] == "typewriter" and len(text) > 1:
        n = len(text)
        events = []
        for k in range(1, n + 1):
            s = start + round(in_ms * (k - 1) / n)
            e = start + round(in_ms * k / n) if k < n else end
            body = escape(text[:k]) + ("{\\alpha&HFF&}" + escape(text[k:]) if k < n else "")
            tags = f"\\an5{pos}" + (out_tags + (f"\\fad(0,{fade_out})" if fade_out else "") if k == n else "")
            events.append(_dialogue(s, e, name, tags, body))
        return events
    return [_dialogue(start, end, name, f"\\an5{pos}{fad}{in_tags}{out_tags}", escape(text))]


def caption_lines(words: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in words:
        if lines and len(lines[-1]) < LINE_WORDS and w["start_ms"] - lines[-1][-1]["end_ms"] <= LINE_GAP_MS:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def replace_words(words: list[dict], new_text: str) -> list[dict]:
    """Spread new words over the original speech time, in proportion to their length."""
    toks = new_text.split()
    if not toks or not words:
        return []
    slots = [(w["start_ms"], w["end_ms"]) for w in words]
    total = sum(e - s for s, e in slots)

    def at(frac: float, is_end: bool) -> int:
        """Map a share of total speech time to a clock time. A boundary between two slots maps to the
        end of the earlier slot for word ends and to the start of the later slot for word starts."""
        target, acc = frac * total, 0
        for s, e in slots:
            if acc + (e - s) > target or (is_end and acc + (e - s) >= target):
                return round(s + (target - acc))
            acc += e - s
        return slots[-1][1]

    lens = [len(t) for t in toks]
    total_chars, done, out = sum(lens), 0, []
    for tok, n in zip(toks, lens):
        out.append({"w": tok, "start_ms": at(done / total_chars, False),
                    "end_ms": at((done + n) / total_chars, True)})
        done += n
    return out


def caption_events(words: list[dict], box: dict) -> list[str]:
    cx, cy = _centre(box)
    events = []
    for line in caption_lines(words):
        parts = []
        for i, w in enumerate(line):
            nxt = line[i + 1]["start_ms"] if i + 1 < len(line) else w["end_ms"]
            parts.append(f"{{\\k{max(1, round((nxt - w['start_ms']) / 10))}}}{escape(w['w'])}")
        events.append(_dialogue(line[0]["start_ms"], line[-1]["end_ms"], "Caption", f"\\an5\\pos({cx},{cy})",
                                " ".join(parts)))
    return events


def build_ass(template: dict, texts: dict[str, str]) -> str | None:
    """The full ASS document, or None when there is nothing to draw."""
    styles, events = [], []
    for item in template["texts"]:
        if item["kind"] != "overlay":
            continue
        ev = overlay_events(item, texts.get(item["slot"], item["sample"]))
        if ev:
            styles.append(style_line(item["id"], item["style"]))
            events += ev
    words = template["captions"]["words"]
    if "captions" in texts:
        words = replace_words(words, texts["captions"])
    if words:
        ref = next((x for x in template["texts"] if x["id"] == template["captions"]["style_ref"]), None)
        styles.append(style_line("Caption", ref["style"] if ref else DEFAULT_CAPTION_STYLE, secondary_alpha=0x80))
        events += caption_events(words, ref["box"] if ref else DEFAULT_CAPTION_BOX)
    if not events:
        return None
    return HEADER.format(w=PLAY_W, h=PLAY_H) + "\n".join(styles) + "\n" + EVENTS + "\n".join(events) + "\n"
