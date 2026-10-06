"""Builds the structured report; saves .txt and .json; formats terminal output."""
import json
from datetime import datetime

from .utils import OUTPUT_DIR, WIDTH, fmt_time
from .vision_analyzer import format_frame

NA = "Not applicable - audio input."


def build_report(info, language, segments, analysis, visual, ocr, engine, device, whisper_model):
    is_video = info["type"] == "video"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_information": {
            "name": info["name"], "path": str(info["path"]), "type": info["type"].title(),
            "extension": info["extension"], "size_mb": info["size_mb"],
            "duration_seconds": info["duration"],
            "duration": fmt_time(info["duration"]) if info["duration"] else "Unknown"},
        "pipeline": {"whisper_model": whisper_model, "device": device, "text_analysis_engine": engine,
                     "vision_mode": (visual or {}).get("mode") if is_video else None},
        "language": language,
        "transcript": [{"start": s["start"], "end": s["end"], "timestamp": fmt_time(s["start"]),
                        "text": s["text"]} for s in segments],
        "main_topic": analysis["main_topic"],
        "summary": analysis["summary"],
        "detailed_summary": analysis["detailed_summary"],
        "key_points": analysis["key_points"],
        "major_topics": analysis["major_topics"],
        "keywords": analysis["keywords"],
        "highlights": [{"timestamp": fmt_time(h["timestamp"]), "seconds": h["timestamp"],
                        "title": h["title"], "description": h["description"]}
                       for h in analysis["highlights"]],
        "sentiment": analysis["sentiment"],
        "entities": analysis["entities"],
        "action_items": analysis["action_items"],
        "visual_analysis": ({"tags": visual.get("tags", []), "frames": [
            {"timestamp": fmt_time(f["timestamp"]), "seconds": f["timestamp"], "description": f["description"]}
            for f in visual.get("frames", [])]} if is_video and visual else
            (None if is_video else NA)),
        "ocr_text": ({"unique_lines": ocr["unique_lines"]} if is_video and ocr else
                     (None if is_video else NA)),
        "multimodal_interpretation": analysis.get("multimodal_interpretation", "") if is_video else NA,
    }


def save_report(report):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt, js = OUTPUT_DIR / "analysis_report.txt", OUTPUT_DIR / "analysis_report.json"
    txt.write_text(format_text(report), encoding="utf-8")
    js.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return txt, js


def _h(title):
    return f"\n{'-' * WIDTH}\n{title}\n{'-' * WIDTH}\n"


def _bullets(items, numbered=False):
    return "\n".join((f"{i}. {x}" if numbered else f"- {x}") for i, x in enumerate(items, 1))


def format_text(r, transcript_limit=None):
    out = ["=" * WIDTH, "AI MULTIMODAL AUDIO & VIDEO ANALYZER - REPORT", "=" * WIDTH]
    f, p = r["file_information"], r["pipeline"]
    out.append(_h("FILE INFORMATION"))
    out.append(f"Input File : {f['name']}\nFile Type  : {f['type']}\nDuration   : {f['duration']}\n"
               f"Size       : {f['size_mb']} MB\nWhisper    : {p['whisper_model']} on {p['device'].upper()}\n"
               f"Analysis   : {p['text_analysis_engine']}")
    lang = r["language"]
    out.append(_h("LANGUAGE"))
    out.append(lang["name"] if lang else "Not detected (no audio transcribed)")

    out.append(_h("TRANSCRIPT"))
    tr = r["transcript"]
    if not tr:
        out.append("No speech detected.")
    else:
        shown = tr if transcript_limit is None else tr[:transcript_limit]
        out.append("\n".join(f"[{s['timestamp']}] {s['text']}" for s in shown))
        if len(shown) < len(tr):
            out.append(f"... ({len(tr) - len(shown)} more segments - see outputs/analysis_report.txt)")

    out.append(_h("VISUAL ANALYSIS"))
    va = r["visual_analysis"]
    if isinstance(va, str):
        out.append(va)
    elif not va:
        out.append("Visual analysis unavailable.")
    else:
        if va["tags"]:
            out.append("Visual Information:\n" + _bullets(va["tags"]) + "\n")
        out.append("\n".join(format_frame({"timestamp": x["seconds"], "description": x["description"]})
                             for x in va["frames"]))
    out.append(_h("OCR TEXT"))
    oc = r["ocr_text"]
    if isinstance(oc, str):
        out.append(oc)
    elif not oc:
        out.append("OCR unavailable.")
    else:
        out.append(_bullets(oc["unique_lines"][:40]) if oc["unique_lines"] else "No text detected.")

    out.append(_h("AI CONTENT ANALYSIS"))
    out.append(f"MAIN TOPIC:\n{r['main_topic']}\n")
    out.append(f"SUMMARY:\n{r['summary'] or 'N/A'}\n")
    out.append(f"DETAILED SUMMARY:\n{r['detailed_summary'] or 'N/A'}\n")
    out.append("KEY POINTS:\n" + (_bullets(r["key_points"], True) or "N/A") + "\n")
    out.append("MAJOR TOPICS:\n" + (_bullets(r["major_topics"]) or "N/A") + "\n")
    out.append("KEYWORDS:\n" + (_bullets(r["keywords"]) or "N/A") + "\n")
    s = r["sentiment"]
    out.append(f"SENTIMENT (of the content, not of any person):\n{s['label']}"
               + (f" - {s['explanation']}" if s["explanation"] else "") + "\n")
    out.append("ENTITIES:")
    out.append("\n".join(f"{k}:\n" + "\n".join(f"  {v}" for v in vals) for k, vals in r["entities"].items())
               or "No clear entities detected.")
    out.append("\nACTION ITEMS:")
    out.append(_bullets(r["action_items"], True) if r["action_items"] else "No clear action items detected.")
    out.append("\nMULTIMODAL INTERPRETATION:")
    out.append(r["multimodal_interpretation"] or "N/A")

    out.append(_h("TIMESTAMPED HIGHLIGHTS"))
    out.append("\n".join(f"{h['timestamp']} - {h['title']}" + (f"\n        {h['description']}" if h["description"] else "")
                         for h in r["highlights"]) or "No highlights available.")
    out.append("\n" + "=" * WIDTH)
    return "\n".join(out)
