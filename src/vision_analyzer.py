"""Frame -> textual description. Swap the model by editing analyze_frame() only."""
import base64

from .utils import (VISION_MODEL, MAX_VISION_FRAMES, api_key_available, pick_evenly, fmt_time)

PROMPT = (
    "Describe this video frame in 1-2 factual sentences. State whether you can see: "
    "people, objects, the scene/setting, a presentation slide, a diagram, a chart or graph, "
    "a computer screen, a demonstration, or readable text (summarise what the text is about). "
    "Describe only what is visible; do not guess."
)

TAGS = {
    "Person detected": ["person", "people", "man", "woman", "speaker", "presenter", "student", "face", "audience"],
    "Presentation slide detected": ["slide", "presentation", "powerpoint", "projector"],
    "Diagram detected": ["diagram", "flowchart", "schematic", "architecture"],
    "Chart/graph detected": ["chart", "graph", "plot", "histogram"],
    "Screen detected": ["screen", "monitor", "laptop", "software", "interface", "code editor", "terminal"],
    "Text detected": ["text", "caption", "title", "heading", "written", "words"],
    "Demonstration detected": ["demonstrat", "experiment", "hands-on", "tutorial"],
}


def _encode(image_path, max_width=768):
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("unreadable image")
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    okay, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def analyze_frame(image_path):
    """Return a text description of one frame (OpenAI vision model)."""
    from openai import OpenAI
    b64 = _encode(image_path)
    resp = OpenAI().chat.completions.create(
        model=VISION_MODEL, max_tokens=160,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
        ]}])
    return resp.choices[0].message.content.strip()


def _local_describe(image_path):
    """Offline fallback: OpenCV face detection only."""
    import cv2
    img = cv2.imread(str(image_path))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    n = len(faces)
    return (f"Basic local analysis: {n} face(s) detected." if n
            else "Basic local analysis: no faces detected.")


def _tags_from(descriptions):
    text = " ".join(d for d in descriptions if "no faces detected" not in d).lower()
    return [tag for tag, words in TAGS.items() if any(w in text for w in words)]


def analyze_frames(frames, notify=print):
    """Analyze a subset of frames. Never raises; falls back to local analysis."""
    chosen = pick_evenly(frames, MAX_VISION_FRAMES)
    results, mode, fails = [], "openai-vision", 0
    if not api_key_available():
        mode = "local-fallback"
        notify("No API key - using basic local visual analysis.")
    for f in chosen:
        desc = None
        if mode == "openai-vision":
            try:
                desc = analyze_frame(f["path"])
                fails = 0
            except Exception as e:
                fails += 1
                if fails >= 2:
                    mode = "local-fallback"
                    notify(f"Vision API unavailable ({str(e)[:80]}) - using local fallback.")
        if desc is None:
            try:
                desc = _local_describe(f["path"])
            except Exception:
                continue
        results.append({"timestamp": f["timestamp"], "description": desc})
    return {"mode": mode, "frames": results, "tags": _tags_from([r["description"] for r in results])}


def format_frame(r):
    return f"Frame at {fmt_time(r['timestamp'])}: {r['description']}"
