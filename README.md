# AI Multimodal Audio & Video Analyzer

## 1. Objective
A command-line backend that takes **any** audio or video file at runtime and automatically produces a structured
analysis report (transcript, summary, key points, topics, keywords, sentiment, entities, action items, timestamped
highlights, visual analysis and on-screen text). No UI, no Q&A, no hard-coded input.

## 2. Architecture
```
INPUT FILE -> type detection -> (video: extract audio) -> WHISPER -> timestamped transcript
                                      |                                   |
                                 video only: frames (OpenCV) -> Vision model + OCR
                                                                          |
                      Transcript + visual descriptions + OCR text -> LLM -> structured report (.txt + .json)
```
| Module | Job |
|---|---|
| `main.py` | Orchestrates the 6 steps, prints the terminal report |
| `src/file_handler.py` | Existence check, audio/video detection, metadata |
| `src/audio_processor.py` | FFmpeg: extract/convert to 16 kHz mono WAV |
| `src/transcription.py` | Whisper speech-to-text with timestamps + language |
| `src/video_processor.py` | OpenCV: representative frames (adaptive sampling) |
| `src/vision_analyzer.py` | Frame -> text description (`analyze_frame()` is the only place to change the vision model) |
| `src/ocr_processor.py` | EasyOCR: visible text in frames |
| `src/text_analyzer.py` | LLM analysis (+ offline fallback if no key / API failure) |
| `src/report_generator.py` | Builds and saves `outputs/analysis_report.txt/.json` |

## 3. Installation (Windows, VS Code terminal, Python 3.10/3.11)
```bat
cd AI_Multimodal_Analyzer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Open `.env` and set `OPENAI_API_KEY=...` (without a key the program still runs with lower-quality offline analysis).
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
Optional NVIDIA GPU: install the CUDA build of PyTorch from pytorch.org; the code uses it automatically, else CPU.

## 4. FFmpeg setup
Easiest: `winget install Gyan.FFmpeg` (or `choco install ffmpeg`). Close and reopen the terminal, then verify:
```bat
ffmpeg -version
```

## 5. How to run
```bat
python main.py "C:\Users\Meghana\Downloads\lecture.mp4"
python main.py "podcast.mp3"
python main.py            (prints usage)
```
First run downloads the Whisper model (~460 MB for `small`) and EasyOCR models; later runs are offline for these.
Tip for the demo: run once beforehand so the models are cached.

## 6. Supported files
Audio: MP3, WAV, M4A, AAC, FLAC, OGG. Video: MP4, MOV, AVI, MKV, WEBM.

## 7-8. Models and what they do
- **Whisper (`small`, local)** - speech -> text with timestamps and language detection. Change with `WHISPER_MODEL` in `.env` (tiny/base/small/medium).
- **Vision model (OpenAI, default `gpt-4o-mini`)** - video frame -> description (people, slides, diagrams, charts, screens). Without API: OpenCV face detection only.
- **EasyOCR** - frame -> visible text (English by default; `OCR_LANGUAGES=en,hi` for more).
- **LLM (default `gpt-4o-mini`)** - transcript + visuals + OCR -> topic, summaries, key points, keywords, sentiment, entities, action items, highlights.
Settings (`FRAME_INTERVAL_SECONDS`, `MAX_FRAMES`, `MAX_VISION_FRAMES`, models) are in `.env` / `src/utils.py`.

## 9. Output
Terminal report plus `outputs/analysis_report.txt` and `outputs/analysis_report.json` (overwritten each run).
Sections: file info, language, transcript, main topic, summaries, key points, topics, keywords, highlights, sentiment,
entities, action items, visual analysis, OCR text. Audio input shows "Not applicable - audio input." for visual sections.
Highlight timestamps are always snapped to real Whisper segment start times.

## 10. Error handling
Readable messages (no tracebacks; set `DEBUG=1` to see one) for: missing file, unsupported format, empty/corrupted file,
missing FFmpeg, no speech, very short audio, Whisper failure (video continues visually), missing key, API failure,
vision failure and OCR failure (each falls back and continues). Long videos are capped at `MAX_FRAMES` frames, and
long transcripts are condensed chunk by chunk before analysis.

## 11. Limitations
- CPU transcription with `small` runs roughly real-time or slower on laptops; long files take a while.
- Vision analyzes at most 8 sampled frames by default, so brief visual events can be missed.
- Sentiment is the tone of the *content*, not a person's real emotional state.
- The offline fallback (no API key) is statistical and much weaker than the LLM analysis.
- Multilingual audio is transcribed, but OCR/analysis quality depends on language settings.

## 12. Future improvements
Speaker diarization, scene-change based frame selection, local vision model (e.g. LLaVA), translation, PDF report export.
