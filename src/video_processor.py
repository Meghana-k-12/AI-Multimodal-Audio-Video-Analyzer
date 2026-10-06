"""Representative frame extraction with OpenCV (never loads the whole video)."""
import math
import shutil

from .utils import (AnalyzerError, TEMP_DIR, FRAME_INTERVAL_SECONDS, MIN_FRAMES, MAX_FRAMES)


def plan_timestamps(duration):
    """Adaptive sampling: ~1 frame per FRAME_INTERVAL_SECONDS, bounded by MIN/MAX_FRAMES."""
    n = math.ceil(duration / FRAME_INTERVAL_SECONDS)
    n = max(MIN_FRAMES, min(n, MAX_FRAMES))
    n = max(1, min(n, int(duration / 0.5)))          # never denser than 2 frames/second
    step = duration / n
    first = min(1.0, duration * 0.1)                 # avoid a black frame at t=0
    times = [first] + [i * step for i in range(1, n)]
    return [min(t, max(duration - 0.1, 0)) for t in times]


def extract_frames(video_path, duration=None):
    """Return [{'timestamp': seconds, 'path': Path}] for sampled frames."""
    try:
        import cv2
    except ImportError:
        raise AnalyzerError("OpenCV is not installed. Run: pip install -r requirements.txt")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise AnalyzerError("OpenCV could not open the video (file may be corrupted).")
    fps, total = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if not duration or duration <= 0:
        if fps > 0 and total > 0:
            duration = total / fps
        else:
            cap.release()
            raise AnalyzerError("Could not determine the video duration.")

    frames_dir = TEMP_DIR / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, t in enumerate(plan_timestamps(duration)):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        okay, img = cap.read()
        if not okay or img is None:
            continue
        h, w = img.shape[:2]
        if w > 1280:
            img = cv2.resize(img, (1280, int(h * 1280 / w)))
        path = frames_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(path), img)
        frames.append({"timestamp": round(t, 2), "path": path})
    cap.release()
    if not frames:
        raise AnalyzerError("No frames could be read from the video.")
    return frames
