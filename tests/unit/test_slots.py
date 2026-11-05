import pytest

from vtpl.render.slots import MediaError, resolve_media

T = {"shots": [{"slot": "clip_1"}, {"slot": "clip_2"}, {"slot": "clip_3"}]}


def touch(p):
    p.write_bytes(b"x")
    return p


def test_by_name_with_reuse_and_music(tmp_path):
    c1 = touch(tmp_path / "clip_1.mp4")
    c3 = touch(tmp_path / "clip_3.MOV")
    m = touch(tmp_path / "music.mp3")
    media = resolve_media(T, tmp_path)
    assert media.clips == {"clip_1": c1, "clip_2": c1, "clip_3": c3}
    assert media.music == m
    assert media.warnings == ["clip_2: no clip found, reusing clip_1.mp4"]


def test_yaml_mapping(tmp_path):
    beach = touch(tmp_path / "beach.mov")
    song = touch(tmp_path / "song.wav")
    (tmp_path / "texts.yaml").write_text(
        "clips:\n  clip_1: beach.mov\n  clip_2: beach.mov\n  clip_3: beach.mov\nmusic: song.wav\ntext_1: HELLO\n")
    media = resolve_media(T, tmp_path)
    assert set(media.clips.values()) == {beach}
    assert media.music == song and media.texts == {"text_1": "HELLO"} and media.warnings == []


def test_missing_first_clip(tmp_path):
    with pytest.raises(MediaError, match="No clip for clip_1"):
        resolve_media(T, tmp_path)


def test_mapped_file_missing(tmp_path):
    (tmp_path / "texts.yaml").write_text("clips:\n  clip_1: nope.mp4\n")
    with pytest.raises(MediaError, match="nope.mp4, which does not exist"):
        resolve_media(T, tmp_path)


def test_no_music_warns(tmp_path):
    touch(tmp_path / "clip_1.mp4")
    media = resolve_media({"shots": [{"slot": "clip_1"}]}, tmp_path)
    assert media.music is None and media.warnings == ["No music file found; output will be silent"]


@pytest.mark.parametrize("yaml_text,msg", [
    ("- clip_1.mp4\n", "texts.yaml must be a mapping"),
    ("clips:\n  - clip_1.mp4\n", "'clips:' in texts.yaml must map slot names to file names"),
    ("clips:\n  clip_1: 5\n", "clips.clip_1 in texts.yaml must be a file name"),
    ("clips:\n  clip_1:\n", "clips.clip_1 in texts.yaml must be a file name"),
    ("music: [a.mp3]\n", "music in texts.yaml must be a file name"),
])
def test_bad_yaml_shapes(tmp_path, yaml_text, msg):
    touch(tmp_path / "clip_1.mp4")
    (tmp_path / "texts.yaml").write_text(yaml_text)
    with pytest.raises(MediaError, match=msg):
        resolve_media({"shots": [{"slot": "clip_1"}]}, tmp_path)


def test_empty_text_values_become_empty_strings(tmp_path):
    touch(tmp_path / "clip_1.mp4")
    (tmp_path / "texts.yaml").write_text("text_1:\ntext_2: ''\ntext_3: 42\n")
    media = resolve_media({"shots": [{"slot": "clip_1"}]}, tmp_path)
    assert media.texts == {"text_1": "", "text_2": "", "text_3": "42"}


def test_invalid_yaml_is_media_error(tmp_path):
    (tmp_path / "texts.yaml").write_text("clips: [unclosed")
    with pytest.raises(MediaError, match="texts.yaml is not valid YAML"):
        resolve_media(T, tmp_path)
