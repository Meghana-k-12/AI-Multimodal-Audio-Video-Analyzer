"""Visible-text extraction from frames with EasyOCR."""
from .utils import MAX_OCR_FRAMES, OCR_LANGUAGES, get_device, pick_evenly


def extract_text(frames):
    """Return {'frames':[{timestamp,lines}], 'unique_lines':[...]}. Raises if OCR unavailable."""
    import easyocr  # ImportError is handled by the caller (optional module)
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=(get_device() == "cuda"), verbose=False)
    per_frame, unique, seen = [], [], set()
    for f in pick_evenly(frames, MAX_OCR_FRAMES):
        try:
            res = reader.readtext(str(f["path"]), detail=1, paragraph=False)
        except Exception:
            continue
        lines = [t.strip() for (_b, t, conf) in res if conf >= 0.4 and len(t.strip()) >= 2]
        per_frame.append({"timestamp": f["timestamp"], "lines": lines})
        for ln in lines:
            if ln.lower() not in seen:
                seen.add(ln.lower())
                unique.append(ln)
    return {"frames": per_frame, "unique_lines": unique}
