"""LLM-based analysis of transcript + visual info, with an offline fallback."""
import json
import re
import time
from collections import Counter

from .utils import LLM_MODEL, api_key_available, fmt_time

CHUNK_CHARS = 6000          # transcripts longer than 2 chunks are condensed first
SENTIMENTS = ("Positive", "Negative", "Neutral", "Mixed")

SYSTEM = (
    "You are a careful content analyst. Use ONLY the material provided. Never invent facts, "
    "names, numbers, entities or action items; if something is absent, return an empty list. "
    "Sentiment describes the tone of the CONTENT only, never a person's real emotional state. "
    "Reply with a single valid JSON object and nothing else.")

SCHEMA = """Return JSON with exactly these keys:
{
 "main_topic": "short phrase",
 "summary": "3-4 sentence concise summary",
 "detailed_summary": "one or two paragraphs",
 "key_points": ["5 to 10 important points"],
 "major_topics": ["main subjects discussed"],
 "keywords": ["8 to 15 important keywords"],
 "sentiment": {"label": "Positive|Negative|Neutral|Mixed", "explanation": "one sentence about the content's tone"},
 "entities": {"People": [], "Organizations": [], "Locations": [], "Technologies": [], "Products": [], "Dates": [], "Numbers": []},
 "action_items": ["only tasks/commitments actually stated; else empty list"],
 "highlights": [{"start_seconds": 0, "title": "short title", "description": "one sentence"}],
 "multimodal_interpretation": "how the visuals/on-screen text relate to what is said (empty string if no visual info)"
}
Rules for highlights: give 4 to 8 items in chronological order. start_seconds MUST be copied exactly
from a bracketed [NNNs] marker in the transcript/notes. Do not make up timestamps."""


class LLMError(Exception):
    pass


def _chat_json(user, max_tokens=2500):
    from openai import OpenAI
    client, last = OpenAI(), None
    for _ in range(2):
        try:
            r = client.chat.completions.create(
                model=LLM_MODEL, temperature=0.2, max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            last = e
            time.sleep(1.5)
    raise LLMError(str(last))


def _transcript_block(segments):
    return "\n".join(f"[{int(s['start'])}s] {s['text']}" for s in segments)


def _condense(segments):
    """Map step for long transcripts: short notes per chunk, keeping chunk start times."""
    chunks, cur, size = [], [], 0
    for s in segments:
        cur.append(s)
        size += len(s["text"]) + 8
        if size >= CHUNK_CHARS:
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)
    notes = []
    for ch in chunks:
        out = _chat_json(
            "Summarise this part of a transcript in about 120 words. Mention named people, "
            "organizations, technologies, dates, numbers and any stated tasks. "
            'Return JSON: {"notes": "..."}\n\n' + _transcript_block(ch), max_tokens=500)
        notes.append(f"[{int(ch[0]['start'])}s] {out.get('notes', '')}")
    return "\n".join(notes)


def _snap(value, starts):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(starts, key=lambda s: abs(s - v)) if starts else None


def _as_list(x):
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [str(x).strip()] if isinstance(x, str) and x.strip() else []


def _normalize(raw, segments):
    starts = [s["start"] for s in segments]
    sent = raw.get("sentiment") if isinstance(raw.get("sentiment"), dict) else {}
    label = str(sent.get("label", "Neutral")).strip().title()
    if label not in SENTIMENTS:
        label = "Neutral"
    entities = {}
    if isinstance(raw.get("entities"), dict):
        for cat, vals in raw["entities"].items():
            vals = _as_list(vals)
            if vals:
                entities[str(cat)] = vals
    highlights, seen = [], set()
    for h in raw.get("highlights", []) if isinstance(raw.get("highlights"), list) else []:
        if not isinstance(h, dict):
            continue
        t = _snap(h.get("start_seconds"), starts)
        if t is None or t in seen:
            continue
        seen.add(t)
        highlights.append({"timestamp": t, "title": str(h.get("title", "")).strip() or "Highlight",
                           "description": str(h.get("description", "")).strip()})
    highlights.sort(key=lambda h: h["timestamp"])
    return {
        "main_topic": str(raw.get("main_topic", "")).strip() or "Not determined",
        "summary": str(raw.get("summary", "")).strip(),
        "detailed_summary": str(raw.get("detailed_summary", "")).strip(),
        "key_points": _as_list(raw.get("key_points"))[:10],
        "major_topics": _as_list(raw.get("major_topics")),
        "keywords": _as_list(raw.get("keywords")),
        "sentiment": {"label": label, "explanation": str(sent.get("explanation", "")).strip()},
        "entities": entities,
        "action_items": _as_list(raw.get("action_items")),
        "highlights": highlights,
        "multimodal_interpretation": str(raw.get("multimodal_interpretation", "")).strip(),
    }


# ----------------------------------------------------------------------------
# Offline fallback (no API key / API failure): simple statistical analysis
# ----------------------------------------------------------------------------
STOP = set("""the a an and or but if then so of to in on at for with from by as is are was were be been being it its
this that these those i you he she we they them his her our your their my me not no yes do does did have has had
will would can could should may might just very really also about into over than too more most some any all one
what which who whom when where why how there here out up down off again so such only own same few both each other
like going get got gonna want need let know think said say says okay well now""".split())
POS = set("good great excellent amazing positive success successful improve improved benefit benefits happy love best "
          "effective helpful easy better growth win".split())
NEG = set("bad poor terrible negative fail failed failure problem problems issue issues worse worst risk difficult hard "
          "loss decline error wrong sad hate".split())
ACTION_RE = re.compile(r"\b(need to|needs to|should|must|have to|has to|will|let's|we'll|i'll|going to|"
                       r"make sure|remember to|please|action item|deadline|submit|schedule|complete)\b", re.I)


def _cap(text):
    return " ".join(w.capitalize() for w in text.split())


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def _offline(segments, has_visual):
    text = " ".join(s["text"] for s in segments)
    if not text.strip():
        return {"main_topic": "Not determined (no speech detected)", "summary": "", "detailed_summary": "",
                "key_points": [], "major_topics": [], "keywords": [],
                "sentiment": {"label": "Neutral", "explanation": "No speech to analyse."},
                "entities": {}, "action_items": [], "highlights": [], "multimodal_interpretation": ""}
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)]
    freq = Counter(w for w in words if w not in STOP)
    keywords = [w for w, _ in freq.most_common(12)]
    sents = _sentences(text) or [text[:300]]
    scored = sorted(range(len(sents)), key=lambda i: -sum(freq.get(w.lower(), 0)
                    for w in re.findall(r"[A-Za-z']+", sents[i])) / (len(sents[i]) ** 0.5))
    pick = lambda n: [sents[i] for i in sorted(scored[:n])]
    pos, neg = sum(w in POS for w in words), sum(w in NEG for w in words)
    label = ("Mixed" if pos > 2 and neg > 2 and min(pos, neg) / max(pos, neg) > 0.6
             else "Positive" if pos > neg * 1.5 and pos > 1 else "Negative" if neg > pos * 1.5 and neg > 1
             else "Neutral")
    caps = Counter(m for m in re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b", text)
                   if m.lower() not in STOP)
    entities = {}
    if caps:
        entities["Possible names/terms"] = [c for c, _ in caps.most_common(8)]
    nums = re.findall(r"\b\d[\d,.]*\s?(?:%|percent|million|billion|thousand)?", text)
    if nums:
        entities["Numbers"] = list(dict.fromkeys(n.strip() for n in nums))[:8]
    actions = [s for s in _sentences(text) if ACTION_RE.search(s)][:6]
    n = min(6, len(segments))
    highlights = []
    for i in range(n):
        lo, hi = i * len(segments) // n, (i + 1) * len(segments) // n
        chunk = segments[lo:hi] or [segments[min(lo, len(segments) - 1)]]
        cw = Counter(w.lower() for s in chunk for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", s["text"])
                     if w.lower() not in STOP)
        title = _cap(", ".join(w for w, _ in cw.most_common(2))) or f"Section {i + 1}"
        highlights.append({"timestamp": chunk[0]["start"], "title": title, "description": chunk[0]["text"][:160]})
    return {"main_topic": _cap(", ".join(keywords[:3])) or "Not determined",
            "summary": " ".join(pick(3)), "detailed_summary": " ".join(pick(7)),
            "key_points": pick(6), "major_topics": [_cap(k) for k in keywords[:5]], "keywords": keywords,
            "sentiment": {"label": label, "explanation": "Estimated from positive/negative word counts."},
            "entities": entities, "action_items": actions, "highlights": highlights,
            "multimodal_interpretation": ""}


# ----------------------------------------------------------------------------
def analyze(segments, visual=None, ocr_lines=None, notify=print):
    """Return (analysis_dict, engine_description). Never raises."""
    visual, ocr_lines = visual or {}, ocr_lines or []
    has_visual = bool(visual.get("frames")) or bool(ocr_lines)
    if not api_key_available():
        notify("No API key found - using the offline fallback analyzer (lower quality).")
        return _offline(segments, has_visual), "offline-fallback (no LLM)"
    try:
        if segments:
            total = sum(len(s["text"]) for s in segments)
            body = (_transcript_block(segments) if total <= CHUNK_CHARS * 2
                    else "(Long recording - condensed notes per section)\n" + _condense(segments))
        else:
            body = "(no speech detected)"
        parts = ["TRANSCRIPT:\n" + body]
        if visual.get("frames"):
            parts.append("VISUAL DESCRIPTIONS OF SAMPLED FRAMES:\n" + "\n".join(
                f"[{fmt_time(f['timestamp'])}] {f['description']}" for f in visual["frames"]))
        if ocr_lines:
            parts.append("TEXT VISIBLE ON SCREEN (OCR):\n" + "\n".join(ocr_lines[:60]))
        parts.append(SCHEMA)
        result = _normalize(_chat_json("\n\n".join(parts)), segments)
        if segments and not result["highlights"]:
            result["highlights"] = _offline(segments, False)["highlights"]
        return result, f"LLM ({LLM_MODEL})"
    except Exception as e:
        notify(f"LLM analysis failed ({str(e)[:100]}). Using offline fallback.")
        return _offline(segments, has_visual), "offline-fallback (LLM failed)"
