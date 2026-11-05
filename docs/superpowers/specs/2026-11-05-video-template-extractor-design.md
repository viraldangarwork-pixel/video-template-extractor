# Video Template Extractor (`vtpl`) — Design

Date: 2026-11-05
Status: Draft for review

## 1. Goal

A local, CPU-only Python command-line tool that:

1. **Analyses** a short-form social video (Reels / Shorts / TikTok, vertical, ≤ 90 s) and writes an editable `template.json` describing its edit: cuts, transitions, beats, text overlays (content, position, style, font, animation), speech captions, camera motion and colour grade.
2. **Renders** a new MP4 from `template.json` plus the user's own clips, text and music, reproducing the original edit.

Non-goals (iteration 1): speed-ramp detection, sticker/emoji detection, GPU or cloud models, a GUI, Remotion export, copying source footage or music into the template.

## 2. Scope (iteration 1)

- Shot detection + transition type (cut / fade / dissolve / flash)
- Beat grid, BPM, on-beat flag per cut, beat pattern
- Speech captions with word timestamps
- Text overlays: detection, tracking, timing refinement, style, font guess, animation type
- Caption vs overlay dedupe
- Camera motion per shot (zoom / pan / static)
- Colour grade per shot (brightness / contrast / saturation)
- `template.json` with JSON Schema validation
- FFmpeg + ASS renderer

## 3. Environment

- Python 3.11+, CPU only.
- External binary: FFmpeg ≥ 6 with libass (`ffmpeg`, `ffprobe` on PATH).
- Python libraries: `scenedetect[opencv]`, `opencv-python-headless`, `numpy`, `librosa`, `soundfile`, `faster-whisper`, `rapidocr-onnxruntime`, `Pillow`, `rapidfuzz`, `jsonschema`, `typer`, `pytest`.
- Fonts: ~40 OFL-licensed Google Fonts bundled in `fonts/`, listed in `fonts/catalog.json`.

## 4. Architecture

A pipeline of independent stages. Each stage is a pure function from input files to one JSON output in `work/<video_id>/`. A stage is skipped if its output exists, unless `--force` is passed. `<video_id>` is the first 12 hex chars of the source file's SHA-256.

All times are integer **milliseconds** from video start. All positions are **normalised** to 0–1 of frame width/height.

| # | Stage | Input | Output |
|---|---|---|---|
| 1 | `ingest` | source video | `norm.mp4` (30 fps CFR, 1080×1920, padded if needed), `audio.wav` (mono 22.05 kHz), `meta.json` |
| 2 | `frames` | `norm.mp4` | `frames/` cache: 540×960 JPEGs at 15 fps + `frames.json` index |
| 3 | `shots` | `norm.mp4`, frames | `shots.json` |
| 4 | `audio` | `audio.wav`, `shots.json` | `audio.json` |
| 5 | `speech` | `audio.wav` | `speech.json` |
| 6 | `text` | frames, full-res `norm.mp4` | `text.json` |
| 7 | `motion` | frames, `shots.json` | `motion.json` |
| 8 | `grade` | frames, `shots.json` | `grade.json` |
| 9 | `merge` | all above | `template.json` |
| 10 | `render` | `template.json`, media folder | `out.mp4`, `render_cmd.txt`, `captions.ass` |

CLI:

```
vtpl analyze <video> [--out DIR] [--force] [--stages a,b,c]
vtpl render <template.json> --media <dir> [--out out.mp4] [--snap-to-beats]
vtpl eval [--golden DIR]
```

## 5. Template format

```json
{
  "version": 1,
  "meta": {"w":1080,"h":1920,"fps":30,"duration_ms":15200,"bpm":124,"source_id":"a1b2c3d4e5f6"},
  "audio": {"beats_ms":[...], "downbeat_hint_ms":[...], "beat_pattern":[2,2,4,2], "music_slot":"music"},
  "shots": [
    {"id":"s1","start_ms":0,"end_ms":1450,"slot":"clip_1","on_beat":true,
     "transition_in":{"type":"cut","duration_ms":0},
     "motion":{"type":"zoom_in","scale_from":1.0,"scale_to":1.15,"dx":0.0,"dy":0.0},
     "grade":{"brightness":0.02,"contrast":1.1,"saturation":1.2}}
  ],
  "texts": [
    {"id":"t1","slot":"text_1","sample":"WAIT FOR IT","kind":"overlay",
     "start_ms":200,"end_ms":1400,
     "box":{"x":0.1,"y":0.7,"w":0.8,"h":0.08},
     "style":{"font":"Montserrat-Bold","font_confidence":0.78,"size_px":72,
              "color":"#FFFFFF","stroke":{"color":"#000000","px":4},"shadow":false},
     "anim":{"in":"pop","in_ms":150,"out":"fade","out_ms":200}}
  ],
  "captions": {"style_ref":"t3","words":[{"w":"hello","start_ms":300,"end_ms":520}]}
}
```

- `slot` values are the placeholders the user fills.
- `kind` is `overlay` or `caption` (see dedupe, §6.8).
- `beat_pattern` = number of beats each shot spans, in order (used by `--snap-to-beats`).
- `font_confidence < 0.5` → `font` is the fallback `Montserrat-Bold` and the original guess is kept in `font_guess`.
- `schema.py` holds the JSON Schema; `merge` validates before writing and `render` validates before reading.

## 6. Algorithms

### 6.1 Ingest
ffprobe for streams. FFmpeg normalises to 30 fps CFR, scales/pads to 1080×1920, extracts mono audio. No audio stream → `meta.has_audio=false`.

### 6.2 Shots and transitions
1. PySceneDetect `AdaptiveDetector` (default threshold 3.0, `min_scene_len` 8 frames) → cut frames.
2. Additionally scan mean luma per frame for fades missed by the detector (luma → < 0.05 or rising from it over ≥ 4 frames).
3. Classify each boundary from the 20 frames around it:
   - **fade**: luma ramps to/from near-black.
   - **flash**: luma spike > 0.85 for ≤ 4 frames.
   - **dissolve**: for frames in the window, each frame ≈ α·A + (1−α)·B (A, B = frames just outside the window) with α monotonic and residual < threshold, over 5–20 frames.
   - otherwise **cut**.
4. Transition `duration_ms` = window length for fade/dissolve/flash, 0 for cut.

### 6.3 Audio
1. `librosa.onset.onset_strength` + `librosa.beat.beat_track` → `bpm`, `beats_ms`.
2. `downbeat_hint_ms`: every 4th beat, phase chosen to maximise summed onset strength.
3. Cut is `on_beat` if within ±80 ms of a beat.
4. `beat_pattern`: per shot, round(shot length / beat period), min 1.
5. Skipped when `has_audio=false` → `bpm=null`, empty arrays.

### 6.4 Speech
faster-whisper `small`, `compute_type="int8"`, `word_timestamps=True`, VAD filter on. Output words with start/end ms and probability. Skipped when no audio.

### 6.5 Text
1. **Detect**: RapidOCR on every 2nd cached frame (15 fps cache → one sample every ≈ 133 ms). If a detected box is < 20 px tall at 540 p, re-run OCR on the full-res frame for that box. Keep boxes with score ≥ 0.6.
2. **Track**: greedy linking frame to frame. Match if IoU ≥ 0.5 **and** `rapidfuzz.ratio ≥ 80` (or text is a prefix — supports typewriter). Allow gaps ≤ 3 samples. Drop tracks shorter than 300 ms. Track text = most frequent string.
3. **Refine timing**: at full frame rate, around start and end ± 200 ms, compute mean absolute pixel difference inside the box vs. the first/last confirmed frame; boundary = first/last frame under threshold.
4. **Style** (on the 3 sharpest frames of the track, full-res crop):
   - Otsu threshold → text mask (pick polarity where mask is the smaller connected-component set inside the box).
   - `color` = median colour of mask pixels.
   - `stroke`: dilate mask by 1–8 px; ring = dilated − mask; if ring colour differs from both fill and background by ΔE > 20, stroke colour = ring median and px = smallest dilation where ring colour stays consistent.
   - `shadow`: offset mask by (2..6, 2..6) px; dark ring present → `true`.
   - `size_px` = cap height in px (from mask rows) × font's units-per-em / cap-height ratio, at 1080 p scale.
5. **Font guess** (analysis by synthesis): for each font in `fonts/catalog.json`, render the track text with Pillow at the measured cap height, binarise, scale to the crop's mask size, score = 0.5·IoU + 0.5·(1 − normalised chamfer distance). Top score → `font`, `font_confidence` = top score. Keep top 3 in `font_candidates`.
6. **Animation** (first and last 300 ms of the track, per frame):
   - box scale curve, centre position curve, text contrast (mask vs. background), OCR character count.
   - **pop**: scale peaks ≥ 1.05× final then settles within 250 ms.
   - **slide**: centre moves ≥ 5 % of frame linearly; direction recorded.
   - **fade**: contrast rises/falls monotonically ≥ 50 % with position and scale stable.
   - **typewriter**: character count increases over ≥ 3 samples.
   - else **none**. Same rules mirrored for `out`.

### 6.6 Motion
Per shot: `cv2.goodFeaturesToTrack` on first frame, `calcOpticalFlowPyrLK` across the shot (re-seed when < 50 points), `estimateAffinePartial2D` (RANSAC) between consecutive frames, accumulate scale and translation. `zoom_in`/`zoom_out` if |total scale − 1| ≥ 0.04; `pan` if |translation| ≥ 3 % of frame; both may apply. < 20 tracked points → `static`.

### 6.7 Grade
Per shot, 5 sampled frames in LAB: mean L, std L, mean chroma. Compared against neutral references (L 0.5, std 0.2, chroma calibrated on synth clips) → `brightness` (−0.3..0.3), `contrast` (0.7..1.5), `saturation` (0.5..1.8), clamped to FFmpeg `eq` ranges.

### 6.8 Merge and dedupe
- Build the single timeline; validate all times inside `[0, duration_ms]`.
- **Dedupe**: a text track is `kind:"caption"` if ≥ 60 % of its words fuzzy-match speech words overlapping its time span. Caption tracks are removed from `texts` (except one kept as `captions.style_ref`) and the speech words become `captions.words`. Otherwise `kind:"overlay"`.
- Assign slots in time order: `clip_1..n`, `text_1..n`, `music`.

## 7. Renderer

Media folder:
```
mine/
  clip_1.mp4 ... clip_n.mp4      # or any names mapped in texts.yaml
  music.mp3
  texts.yaml                     # text_1: "MY TEXT", captions: "new spoken line ...", clips: {clip_1: beach.mov}
```

1. **Slot mapping**: clips by name or by `texts.yaml` mapping; missing clip → reuse previous slot's clip (warning). Missing text → keep `sample`.
2. **Per shot chain**: `trim` (centre of source, shot length / speed) → `setpts` → `scale`+`crop` to 1080×1920 → `zoompan` (scale_from→scale_to, dx/dy) → `eq`.
3. **Join**: `concat` for cuts; `xfade` (`fade`, `dissolve`, `fadeblack`) with detected duration; flash = `xfade=fadewhite` with 2–4 frame duration.
4. **Text**: one ASS file; styles from `style` (font, size, primary colour, outline colour/width, shadow). Animations:
   - pop: `{\fscx80\fscy80\t(0,100,\fscx115\fscy115)\t(100,150,\fscx100\fscy100)}`
   - slide: `{\move(x1,y1,x2,y2,0,in_ms)}`
   - fade: `{\fad(in_ms,out_ms)}`
   - typewriter: one Dialogue event per character prefix, evenly spaced over `in_ms`.
   Positions from `box` → `\pos` / `\an` alignment. Burned in with `subtitles=captions.ass:fontsdir=fonts/`.
5. **Captions**: karaoke `\k` events using `captions.style_ref` style; replacement text spreads timing across new words proportionally to word length.
6. **Audio**: music trimmed to `duration_ms`, 300 ms `afade` out. `--snap-to-beats`: run §6.3 on the user's music and rescale each shot to `beat_pattern[i]` × new beat period.
7. **Output**: `libx264 -crf 20 -preset medium`, `aac 192k`, `-movflags +faststart`. Full command written to `render_cmd.txt`.

## 8. Error handling

- Preflight in CLI: FFmpeg/ffprobe present and built with libass; input readable; fail with a message stating cause and fix.
- Graceful degradation (stage writes a valid empty/neutral result and a warning, never aborts the run):

| Condition | Result |
|---|---|
| No audio stream | `bpm:null`, no beats, no speech |
| No text found | `texts: []` |
| Font confidence < 0.5 | fallback font, `font_guess` kept |
| < 20 motion points | `motion.type:"static"` |
| OCR model load fails | text stage marked failed in `run.json`, template still written |

- Each stage logs to `work/<id>/<stage>.log`; `run.json` records per-stage status, duration and warnings.
- Schema validation errors name the JSON path and expected type.

## 9. Testing

1. **Synthetic fixtures** (`tests/synth/make_fixtures.py`, generated with FFmpeg, ground truth in a sidecar JSON): colour-card cuts at known times, fades/dissolves/flashes, 120 BPM click track, `drawtext` overlays with known fonts and pop/fade/slide/typewriter, zoompan shots, `eq`-graded shots, TTS-free speech fixture using a short bundled public-domain clip.
2. **Tolerances**: cuts ±1 frame; beats ±30 ms; text start/end ±100 ms; correct font in top 3; correct animation label; motion type correct; grade values ±0.1.
3. **Round trip**: analyse fixture → render with its own template → analyse render → templates equal within tolerances.
4. **Unit tests**: IoU, track linking, animation classifier, beat_pattern, dedupe, ASS line builder, filter-graph builder, schema.
5. **Golden set** (`vtpl eval`, manual, not in git): 3–5 real Reels with hand-checked templates; reports per-field accuracy.

## 10. Project layout

```
vtpl/
  cli.py  ingest.py  frames.py  merge.py  schema.py  util/{ffmpeg.py,timebase.py,log.py}
  analyzers/
    shots.py audio.py speech.py motion.py grade.py
    text/{detect.py,track.py,refine.py,style.py,font.py,anim.py}
  render/{slots.py,graph.py,ass.py,run.py}
fonts/  catalog.json  *.ttf
tests/  synth/  unit/  roundtrip/
pyproject.toml  README.md
```

## 11. Risks

- Font guess limited to the bundled catalogue.
- OCR misses on motion-blurred or stylised text → tracking tolerates gaps; animation may read `none`.
- Whisper word timing ±100–500 ms → karaoke drift; forced alignment deferred.
- `librosa` beat positions weaker than Beat This; swap-in point is `analyzers/audio.py`.
- Processing time on CPU: target ≤ 3 min for a 30 s clip on a 4-core laptop; OCR dominates.
