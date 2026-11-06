# vtpl — video template extractor

Turns a short-form video (Reels / Shorts / TikTok) into an editable `template.json`, then renders a new video from that template with your own clips and music. Runs locally on CPU.

## Install

Needs Python 3.11+ and FFmpeg 6+ with libass.

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
```

## Use

```bash
vtpl analyze reel.mp4                 # writes work/<id>/template.json
vtpl render work/<id>/template.json --media ./mine --out out.mp4
vtpl render work/<id>/template.json --media ./mine --snap-to-beats
```

`./mine` holds `clip_1.mp4 … clip_N.mp4` and `music.mp3`. Optional `texts.yaml`:

```yaml
clips: {clip_1: beach.mov}
music: song.wav
```

Text overlays and captions found in the source are in `template.json` under `texts` and `captions`.
Replace them in `texts.yaml` (missing keys keep the original text; an empty value hides it):

```yaml
text_1: MY HEADLINE
text_2: ""            # hide this overlay
captions: new spoken line to show as karaoke captions
```

Captions are burned only when the source showed them (`captions.style_ref` names the caption
text in `texts`) or when `texts.yaml` has a `captions:` line; speech alone draws nothing. A
`captions:` line is spread over the original speech timing, so the template needs speech words.

Fonts are matched against the bundled set in `vtpl/assets/fonts` (see its README for licences).

Speech uses faster-whisper `small` by default; set `VTPL_WHISPER_MODEL=tiny` for speed. The first run downloads the model.
To skip speech (and the Whisper download) entirely, run only the other stages:
`vtpl analyze reel.mp4 --stages ingest,shots,audio,merge` (the template then has no captions).

## Test

```bash
pytest                 # all tests
pytest -m "not slow"   # unit tests only, no FFmpeg
```
