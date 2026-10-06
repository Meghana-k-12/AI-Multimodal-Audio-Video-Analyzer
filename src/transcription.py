"""Speech recognition with OpenAI Whisper (local model)."""
from .utils import AnalyzerError


def load_model(model_name, device):
    try:
        import whisper
    except ImportError:
        raise AnalyzerError("Whisper is not installed. Run: pip install -r requirements.txt")
    try:
        return whisper.load_model(model_name, device=device)
    except Exception as e:
        raise AnalyzerError(f"Could not load Whisper model '{model_name}': {e}")


def transcribe(audio_path, model_name, device):
    """Return {'language','language_name','text','segments':[{start,end,text}]}."""
    model = load_model(model_name, device)
    try:
        try:
            result = model.transcribe(str(audio_path), fp16=(device == "cuda"), verbose=False)
        except RuntimeError:
            if device != "cuda":
                raise
            model = load_model(model_name, "cpu")      # GPU problem -> CPU fallback
            result = model.transcribe(str(audio_path), fp16=False, verbose=False)
    except AnalyzerError:
        raise
    except Exception as e:
        raise AnalyzerError(f"Whisper transcription failed: {e}")

    segments = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if text:
            segments.append({"start": round(float(s["start"]), 2),
                             "end": round(float(s["end"]), 2), "text": text})
    code = result.get("language", "unknown")
    try:
        from whisper.tokenizer import LANGUAGES
        name = LANGUAGES.get(code, code).title()
    except Exception:
        name = code
    return {"language": code, "language_name": name,
            "text": " ".join(s["text"] for s in segments), "segments": segments}
