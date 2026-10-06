"""Shared configuration and small helpers."""
import os
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv missing: continue with plain env vars
    def load_dotenv(*_a, **_k):
        return False

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ---------------- Configuration (override in .env, no code change needed) ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", LLM_MODEL)
FRAME_INTERVAL_SECONDS = float(os.getenv("FRAME_INTERVAL_SECONDS", "10"))
MIN_FRAMES = 5          # short videos are sampled more densely than the interval
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "40"))
MAX_VISION_FRAMES = int(os.getenv("MAX_VISION_FRAMES", "8"))
MAX_OCR_FRAMES = int(os.getenv("MAX_OCR_FRAMES", "20"))
OCR_LANGUAGES = [x.strip() for x in os.getenv("OCR_LANGUAGES", "en").split(",") if x.strip()]

TEMP_DIR = ROOT_DIR / "temp"
OUTPUT_DIR = ROOT_DIR / "outputs"
WIDTH = 60


class AnalyzerError(Exception):
    """A user-readable error that stops the pipeline."""


def fmt_time(seconds):
    seconds = int(max(0, seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def api_key_available():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and key != "your_key_here"


def pick_evenly(items, n):
    """Pick at most n items spread evenly across the list."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / (n - 1) if n > 1 else 0
    return [items[round(i * step)] for i in range(n)]


# ---------------- Terminal helpers ----------------
def banner(title):
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def step(n, total, text):
    print(f"\n[{n}/{total}] {text}")


def ok(text):
    print(f"  ✓ {text}")


def warn(text):
    print(f"  ! {text}")
