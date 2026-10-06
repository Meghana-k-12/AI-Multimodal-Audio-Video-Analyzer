"""AI Multimodal Audio & Video Analyzer - command line entry point.

Usage:  python main.py <audio_or_video_file>
"""
import os
import sys
import time

from src import (audio_processor, file_handler, ocr_processor, report_generator,
                 text_analyzer, transcription, video_processor, vision_analyzer)
from src.utils import (AnalyzerError, WHISPER_MODEL, banner, ffmpeg_available, fmt_time,
                       get_device, ok, step, warn)

USAGE = '''Usage:
python main.py <audio_or_video_file>

Example:
python main.py "lecture.mp4"'''
TOTAL = 6


def run(path_arg):
    started = time.time()
    # ---------------- 1. Detect input ----------------
    step(1, TOTAL, "Detecting input...")
    info = file_handler.inspect_file(path_arg)
    is_video = info["type"] == "video"
    ok(f"{info['type'].title()} detected ({info['extension']}, {info['size_mb']} MB)")
    print(f"\nInput File : {info['name']}\nFile Type  : {info['type'].title()}\n"
          f"Duration   : {fmt_time(info['duration']) if info['duration'] else 'Unknown'}")

    # ---------------- 2. Audio ----------------
    step(2, TOTAL, "Extracting audio..." if is_video else "Preparing audio...")
    if not ffmpeg_available():
        raise AnalyzerError("FFmpeg was not found. Install it and check 'ffmpeg -version' (see README).")
    audio_path = audio_processor.extract_audio(info["path"], info["type"])
    if audio_path is None:
        warn("This video has no audio track - continuing with visual analysis only.")
    else:
        ok("Audio extracted and converted to 16 kHz mono WAV")

    # ---------------- 3. Whisper ----------------
    device = get_device()
    step(3, TOTAL, f"Transcribing with Whisper ({WHISPER_MODEL}, {device.upper()})...")
    language, segments = None, []
    if audio_path is None:
        warn("Skipped - no audio to transcribe.")
    else:
        try:
            print("  (first run downloads the Whisper model - please wait)")
            result = transcription.transcribe(audio_path, WHISPER_MODEL, device)
            segments = result["segments"]
            language = {"code": result["language"], "name": result["language_name"]}
            ok("Transcription completed")
            print(f"  Detected language: {language['name']}")
            if not segments:
                warn("No speech detected in the audio.")
            elif info["duration"] and info["duration"] < 3:
                warn("Very short audio - results may be limited.")
        except AnalyzerError as e:
            if not is_video:
                raise
            warn(f"{e}\n  Continuing with visual analysis only.")

    # ---------------- 4. Video frames / vision / OCR ----------------
    step(4, TOTAL, "Extracting video frames..." if is_video else "Video analysis (skipped for audio input)")
    visual, ocr, frames = None, None, []
    if is_video:
        try:
            frames = video_processor.extract_frames(info["path"], info["duration"])
            ok(f"{len(frames)} representative frames extracted")
        except AnalyzerError as e:
            warn(f"Frame extraction failed: {e}")
        if frames:
            try:
                visual = vision_analyzer.analyze_frames(frames, notify=warn)
                ok(f"Visual analysis completed ({visual['mode']}, {len(visual['frames'])} frames)")
            except Exception as e:
                warn(f"Visual analysis skipped: {str(e)[:100]}")
            try:
                ocr = ocr_processor.extract_text(frames)
                ok(f"OCR completed ({len(ocr['unique_lines'])} unique text lines)")
            except ImportError:
                warn("EasyOCR not installed - continuing without OCR.")
            except Exception as e:
                warn(f"OCR skipped: {str(e)[:100]}")
    else:
        ok("Audio-only input: visual sections not applicable")

    # ---------------- 5. AI analysis ----------------
    step(5, TOTAL, "Performing AI analysis...")
    analysis, engine = text_analyzer.analyze(
        segments, visual, (ocr or {}).get("unique_lines", []), notify=warn)
    ok(f"Analysis completed using {engine}")

    # ---------------- 6. Report ----------------
    step(6, TOTAL, "Generating report...")
    report = report_generator.build_report(info, language, segments, analysis, visual, ocr,
                                           engine, device, WHISPER_MODEL)
    txt, js = report_generator.save_report(report)
    ok(f"Report saved: {txt}")
    ok(f"JSON saved:   {js}")

    print()
    banner("ANALYSIS COMPLETE")
    print(report_generator.format_text(report, transcript_limit=25))
    print(f"\nFinished in {time.time() - started:.1f} seconds.")


def main(argv):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(USAGE)
        return 0
    banner("AI MULTIMODAL AUDIO & VIDEO ANALYZER")
    try:
        run(" ".join(argv[1:]))
        return 0
    except AnalyzerError as e:
        print(f"\nERROR: {e}")
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {type(e).__name__}: {e}")
        if os.getenv("DEBUG") == "1":
            raise
        print("(Set environment variable DEBUG=1 to see the full traceback.)")
    finally:
        audio_processor.cleanup_temp()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
