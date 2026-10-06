"""Input validation, type detection and metadata."""
import shutil
import subprocess
from pathlib import Path

from .utils import AnalyzerError

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _probe_duration(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def _cv_duration(path):
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames / fps if fps and frames and fps > 0 else None
    except Exception:
        return None


def inspect_file(path_str):
    """Validate the file and return a metadata dict."""
    path = Path(str(path_str).strip().strip('"').strip("'")).expanduser()
    if not path.exists():
        raise AnalyzerError(f"File not found: {path}")
    if not path.is_file():
        raise AnalyzerError(f"Not a file: {path}")
    ext = path.suffix.lower()
    if ext in AUDIO_EXT:
        kind = "audio"
    elif ext in VIDEO_EXT:
        kind = "video"
    else:
        raise AnalyzerError(
            f"Unsupported file format '{ext or 'none'}'.\n"
            f"  Audio: {', '.join(sorted(AUDIO_EXT))}\n  Video: {', '.join(sorted(VIDEO_EXT))}")
    size = path.stat().st_size
    if size == 0:
        raise AnalyzerError("The file is empty (0 bytes) - it may be corrupted.")
    duration = _probe_duration(path)
    if duration is None and kind == "video":
        duration = _cv_duration(path)
    return {"path": path, "name": path.name, "extension": ext, "type": kind,
            "size_mb": round(size / (1024 * 1024), 2), "duration": duration}
