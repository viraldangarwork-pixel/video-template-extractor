# Video Template Extractor — Research & Plan

## Goal
Local Python (CPU-only) tool that reverse-engineers a short-form social video (Reels/Shorts/TikTok) into an editable `template.json` (cuts, beats, text overlays with style + animation, motion, grade, captions) and re-renders a new MP4 from that template with the user's own media.

Target users: creators / social media teams who want to "copy the edit" of a trending video with their own clips.

## Market Research
- **CapCut**: Auto Cut / Beat Sync analyses music and places beat markers, then cuts clips to beats; templates let users drop clips into fixed slots. It does **not** (publicly) turn an arbitrary finished video into a template — that gap is this tool's value.
- **Opus Clip / similar AI tools**: focus on auto-cutting long video into clips + beat sync, not on reproducing a reference video's edit.
- **User expectations**: cuts must land on beats (±1–3 frames feels "off"), text must look like the original (font weight, stroke, pop animation), templates must be slot-based ("drop 6 clips here").
- **Compliance**: tool extracts *structure* (timing, layout, style), not the source footage/music. Output template must not embed source media; music slot is user-supplied (copyright).

## Technical Options
| Building block | Option | Pros | Cons | Fit |
|---|---|---|---|---|
| Shot cuts | PySceneDetect AdaptiveDetector | CPU-only, fast, top hard-cut F1 on its own benchmarks | Weak on gradual transitions | **MVP** |
| | TransNetV2 (PyTorch, CPU ok) | Higher accuracy, handles dissolves | Slower, extra model | Later (optional) |
| Beats | librosa beat_track + onset | Simple, light | Beat positions less accurate | **MVP** |
| | Beat This (2024) | SOTA beats + downbeats, no DBN | PyTorch, heavier | Nice-to-have |
| | madmom DBN | Strong accuracy | Old, install pain on new Python | Skip |
| Speech | faster-whisper small int8 | ~4–6x faster than whisper.cpp on x86 CPU, word timestamps | Word timing ±100–500 ms | **MVP** |
| | + wav2vec2 forced alignment | ~±30 ms word timing | Extra model | Later |
| OCR | RapidOCR (PaddleOCR models via ONNX) | Paddle accuracy, light install, CPU | Less docs | **MVP** |
| | EasyOCR | Fast per image | Pulls full PyTorch, lower accuracy | Skip |
| Font guess | Analysis-by-synthesis vs ~40 popular social fonts | No training, explainable, CPU | Limited to font library | **MVP** |
| | DeepFont-style CNN | General | No maintained Latin model; needs training | Later |
| Render | FFmpeg filter_complex + ASS subtitles (libass) | One binary, ASS supports fade/move/scale/karaoke | Complex graph strings | **MVP** |
| | Remotion | Rich animation | Node runtime (user chose Python-only) | Skip |

## Feature Suggestions
### Must-have (MVP)
- **Shot detection** — cut times + slots / why: core of any template / S / PySceneDetect Adaptive
- **Beat grid + on-beat snapping** — beats, BPM, flag cuts within ±80 ms of a beat / why: users expect beat-synced edits / S / librosa
- **Text overlay tracking** — OCR on sampled frames, IoU+text tracking into tracks with start/end / M / RapidOCR
- **Text style extraction** — fill colour, stroke, size, position / M / OpenCV (Otsu + k-means)
- **Font guess** — match crop against rendered candidate fonts / M / Pillow + similarity
- **Text animation classify** — pop / slide / fade / typewriter from box scale, position, contrast, char count over time / M / rules
- **Speech captions** — word timestamps / S / faster-whisper
- **Template JSON + renderer** — FFmpeg + ASS / L

### Nice-to-have
- **Camera motion per shot** — zoom/pan via feature-track affine fit / M / OpenCV
- **Colour grade approximation** — LAB stats → FFmpeg `eq` / S
- **Transition type** — cut / fade / dissolve / flash (luma ramp + histogram blend) / S
- **Caption vs overlay dedupe** — OCR text matching speech words → auto-caption style / S

### Advanced / later
- Speed-ramp detection, TransNetV2, Beat This, forced alignment, sticker/emoji detection, CNN font recogniser, Remotion export.

## Recommended Scope (Iteration 1)
MVP + transition type + caption dedupe — both cheap and noticeably improve template fidelity.

## Risks & Open Questions
- Font guess limited to bundled font library; unknown fonts fall back to nearest match.
- OCR on stylised/animated text (motion blur, emoji) will miss some frames → tracking must tolerate gaps.
- Whisper word timing drift (±100–500 ms) may misalign karaoke captions.
- Speed ramps from a single video are ambiguous → deferred.

## Sources
- [CapCut — How to use Auto Cut](https://www.capcut.com/help/how-to-use-auto-cut)
- [Opus — Best AI beat-sync tools](https://www.opus.pro/blog/best-ai-beat-sync)
- [PySceneDetect benchmarks](https://www.scenedetect.com/benchmarks/)
- [CineTrans (arXiv 2508.11484) — PySceneDetect vs TransNetV2](https://arxiv.org/pdf/2508.11484)
- [transnetv2-pytorch](https://pypi.org/project/transnetv2-pytorch/)
- [Codesota — PaddleOCR vs EasyOCR](https://codesota.com/ocr/paddleocr-vs-easyocr)
- [Codesota — Best OCR for Python](https://codesota.com/ocr/best-for-python)
- [faster-whisper (SYSTRAN)](https://github.com/SYSTRAN/faster-whisper)
- [Beat This / beat tracking evaluation (arXiv 2407.21658)](https://arxiv.org/html/2407.21658v1)
- [DeepFont paper](https://research.adobe.com/publication/deepfont-identify-your-font-from-an-image)
