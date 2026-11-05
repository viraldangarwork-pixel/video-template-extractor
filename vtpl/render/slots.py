from dataclasses import dataclass, field
from pathlib import Path

import yaml

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


class MediaError(ValueError):
    pass


@dataclass
class Media:
    clips: dict[str, Path]
    music: Path | None
    texts: dict[str, str]
    warnings: list[str] = field(default_factory=list)


def _existing(media_dir: Path, name: str, slot: str) -> Path:
    p = media_dir / name
    if not p.exists():
        raise MediaError(f"{slot} is mapped to {name}, which does not exist in {media_dir}")
    return p


def resolve_media(template: dict, media_dir: Path) -> Media:
    media_dir = Path(media_dir)
    cfg_path = media_dir / "texts.yaml"
    cfg = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    mapping = cfg.get("clips") or {}
    by_stem = {p.stem: p for p in sorted(media_dir.iterdir()) if p.suffix.lower() in VIDEO_EXT}

    clips, warnings, prev = {}, [], None
    for shot in template["shots"]:
        slot = shot["slot"]
        path = _existing(media_dir, mapping[slot], slot) if slot in mapping else by_stem.get(slot)
        if path is None:
            if prev is None:
                raise MediaError(f"No clip for {slot}. Add {slot}.mp4 to {media_dir} or map it under 'clips:' in texts.yaml")
            warnings.append(f"{slot}: no clip found, reusing {prev.name}")
            path = prev
        clips[slot] = prev = path

    if cfg.get("music"):
        music = _existing(media_dir, cfg["music"], "music")
    else:
        music = next((p for p in sorted(media_dir.iterdir()) if p.stem == "music" and p.suffix.lower() in AUDIO_EXT), None)
    if music is None:
        warnings.append("No music file found; output will be silent")

    texts = {k: str(v) for k, v in cfg.items() if str(k).startswith("text_")}
    return Media(clips, music, texts, warnings)
