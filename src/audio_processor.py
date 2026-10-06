"""Audio extraction/conversion with FFmpeg (16 kHz mono WAV for Whisper)."""
import shutil
import subprocess

from .utils import AnalyzerError, TEMP_DIR


def extract_audio(src, kind):
    """Return path to a Whisper-ready WAV, or None if a video has no audio track."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AnalyzerError(
            "FFmpeg was not found. Install it and make sure 'ffmpeg -version' works "
            "in a new terminal (see README, section 'FFmpeg setup').")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out = TEMP_DIR / "extracted_audio.wav"
    if out.exists():
        out.unlink()
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src), "-vn",
           "-ac", "1", "-ar", "16000", "-f", "wav", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    err = (r.stderr or "").strip()
    if r.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
        low = err.lower()
        if kind == "video" and ("does not contain any stream" in low or "matches no streams" in low):
            return None
        last = err.splitlines()[-1] if err else "unknown error"
        raise AnalyzerError(f"Could not read audio (file may be corrupted). FFmpeg said: {last}")
    return out


def cleanup_temp():
    wav = TEMP_DIR / "extracted_audio.wav"
    try:
        if wav.exists():
            wav.unlink()
        shutil.rmtree(TEMP_DIR / "frames", ignore_errors=True)
    except Exception:
        pass
