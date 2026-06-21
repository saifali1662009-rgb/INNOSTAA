"""
research.py  —  INNOSTAA Research Mode Module (v5)
══════════════════════════════════════════════════════════════════════════════
What's new in v5:
  ✦ FULLY AI-DRIVEN — no more dumb keyword stripping that misreads context
  ✦ AI reads the full user sentence and decides:
        • what topic to actually search
        • what angle/framing to use (e.g. "USA's PRO-AI arguments for MUN")
        • what additional clarification to ask (if any)
        • what report type to produce
  ✦ AI generates its own search queries — multiple if needed
  ✦ New report type: speech/debate (MUN speeches, debate prep, position papers)
  ✦ Multi-query search — searches 2-4 focused queries and merges results
  ✦ Zero hardcoded topic parsing — the AI understands context, not just nouns
  ✦ Cleaner follow-up: only asks when genuinely needed, not every single time

Integration (unchanged from v4):
    import research
    if 'research mode' in intent:
        research.start(user_text, speak=speak, listen=listen)
        return
"""

import os, re, io, tempfile, requests
from datetime import datetime
from bs4 import BeautifulSoup
from ddgs import DDGS
from groq import Groq
from conversation_history import history
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, PageBreak, Image as RLImage
)
from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.3-70b-versatile"

MAX_RESULTS   = 6
MAX_CHARS     = 10000   # was 18000 — less raw text = tighter, focused report
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
PAGE_W        = A4[0] - 40 * mm
IMG_MAX_W     = 130 * mm
IMG_MAX_H     = 90  * mm

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Cancellation ───────────────────────────────────────────────────────────────
# Set to True by cancel() to abort a running research session cleanly.
# Reset to False at the start of every new start() call.
_CANCELLED = False

_CANCEL_WORDS = {
    "stop", "cancel", "abort", "quit", "exit", "no", "never mind",
    "nevermind", "forget it", "forget this", "leave it", "leave this",
    "don't research", "do not research", "stop research", "stop it",
    "stop researching", "cancel research", "drop it", "skip it",
    "i changed my mind", "changed my mind", "not now", "not anymore",
    "used you accidently", "used accidentally", "by accident", "accident",
}


def cancel():
    """Call this from outside (e.g. voice command) to abort a running research."""
    global _CANCELLED
    _CANCELLED = True
    print("[research] Cancellation requested.")


def _is_cancel(text: str) -> bool:
    """Return True if the user's text is a cancel/stop signal."""
    t = text.lower().strip().rstrip(".")
    # exact match
    if t in _CANCEL_WORDS:
        return True
    # substring match for short inputs
    for word in _CANCEL_WORDS:
        if len(word) > 4 and word in t:
            return True
    return False


def _check_cancel(speak) -> bool:
    """
    Check the global cancel flag. If set, speak a cancellation message,
    reset the flag, and return True so the caller can `return` immediately.
    """
    global _CANCELLED
    if _CANCELLED:
        _CANCELLED = False
        speak("Research cancelled. Let me know if you need anything else.")
        return True
    return False


# ── Colours ────────────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#1A1A2E")
C_ACCENT  = colors.HexColor("#E94560")
C_MID     = colors.HexColor("#16213E")
C_WHITE   = colors.white
C_TEXT    = colors.HexColor("#1A1A2E")
C_SUBTLE  = colors.HexColor("#777777")
C_EQ_BG   = colors.HexColor("#EEF0FF")
C_IMG_BDR = colors.HexColor("#DDDDDD")
C_FORMULA = colors.HexColor("#F8F9FF")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — AI Intent Parser
#  This replaces ALL the old keyword-stripping logic.
#  The AI reads the raw user sentence and figures out EVERYTHING.
# ══════════════════════════════════════════════════════════════════════════════

def _ai_parse_intent(user_text: str) -> dict:
    """
    Ask the AI to fully understand what the user wants to research.

    Returns a dict with:
      - display_topic   : clean topic name for the cover page / speech
      - search_queries  : list of 2-4 search queries to run
      - report_type     : formula_sheet | study_notes | timeline | analysis |
                          speech_debate | general
      - framing         : extra context for the AI writer
                          (e.g. "Write from the USA's perspective, pro-AI stance")
      - needs_clarify   : bool — does the AI need one more question from user?
      - clarify_question: the question to ask (if needs_clarify is True)

    CRITICAL: This function NEVER strips words from the user's input.
    The AI reads the full sentence and understands it semantically.
    """
    recent_context = _history_context(10)
    prompt = f"""You are INNOSTAA's Research Intent Analyser.

The user said: "{user_text}"

Your job is to understand EXACTLY what they want researched and how.
You must read the FULL sentence — do NOT just grab the last noun.

Examples of correct interpretation:
- "MUN topic: Points USA should speak about AI in its favour"
  → display_topic: "USA Position on AI — MUN Speech"
  → search_queries: ["USA pro-AI policy arguments 2024", "artificial intelligence benefits USA economy", "US government AI strategy", "AI regulation USA stance MUN"]
  → report_type: speech_debate
  → framing: "Write arguments from USA's perspective supporting AI. This is for a MUN speech."

- "make a formula sheet for derivatives"
  → display_topic: "Derivatives — Formula Sheet"
  → search_queries: ["derivatives formulas calculus complete list", "differentiation rules all formulas"]
  → report_type: formula_sheet
  → framing: "List every derivative formula exhaustively."

- "research climate change causes and effects"
  → display_topic: "Climate Change: Causes and Effects"
  → search_queries: ["climate change main causes scientific", "climate change effects 2024", "global warming consequences"]
  → report_type: analysis
  → framing: "Comprehensive scientific analysis of causes and effects."

- "study notes on World War 2"
  → display_topic: "World War 2 — Study Notes"
  → search_queries: ["World War 2 key events summary", "WW2 causes major battles outcome"]
  → report_type: study_notes
  → framing: "Concise study notes format."
above are only the example, shows how to think, user can ask for anything.
the history of the session is: {recent_context} use this memory use to analyse which topic user asking to research, if the topic is not mention directly or in-directly or if you didn't understand ask follow-up!! 
Now analyse: "{user_text}"

Respond with ONLY a valid JSON object, no markdown, no explanation:
{{
  "display_topic": "...",
  "search_queries": ["query1", "query2", "query3"],
  "report_type": "one of: formula_sheet | study_notes | timeline | analysis | speech_debate | general",
  "framing": "...",
  "needs_clarify": false,
  "clarify_question": ""
}}

Rules:
- search_queries: 2 to 4 queries. Make them specific and targeted. Include the angle/framing in the queries.
- If the user clearly stated what they want, set needs_clarify to false.
- Only set needs_clarify to true if the request is genuinely too vague to research (e.g. just said "research" with no topic).
- report_type speech_debate: for MUN speeches, debate prep, position papers, persuasive arguments, policy advocacy.
- report_type analysis: for deep analytical reports, comparisons, impact studies.
- report_type formula_sheet: ONLY when user explicitly wants formulas/equations/rules listed.
- report_type study_notes: for concise notes, quick revision, key points.
- report_type timeline: for historical chronological content.
- report_type general: for everything else."""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if model adds them
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        import json
        result = json.loads(raw)
        # Validate required fields
        result.setdefault("display_topic",    user_text.strip().title())
        result.setdefault("search_queries",   [user_text.strip()])
        result.setdefault("report_type",      "general")
        result.setdefault("framing",          "")
        result.setdefault("needs_clarify",    False)
        result.setdefault("clarify_question", "")
        # Ensure report_type is valid
        valid_types = ("formula_sheet","study_notes","timeline","analysis","speech_debate","general")
        if result["report_type"] not in valid_types:
            result["report_type"] = "general"
        print(f"[research] AI parsed intent: topic='{result['display_topic']}' "
              f"type='{result['report_type']}' queries={result['search_queries']}")
        return result
    except Exception as e:
        print(f"[research] Intent parse error: {e}")
        # Graceful fallback — still does something sensible
        return {
            "display_topic":    user_text.strip().title(),
            "search_queries":   [user_text.strip()],
            "report_type":      "general",
            "framing":          "",
            "needs_clarify":    False,
            "clarify_question": ""
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Optional Follow-Up (only when AI says it's needed)
# ══════════════════════════════════════════════════════════════════════════════

def _maybe_ask_clarification(intent: dict, speak, listen) -> dict:
    """
    If the AI flagged the request as too vague, ask the one clarifying question.
    The answer is fed back through the AI parser for a fresh intent.
    """
    if not intent.get("needs_clarify"):
        return intent

    q = intent.get("clarify_question", "Could you tell me more about what you need?")
    speak(q)
    answer = (listen() or "").strip()
    if not answer or len(answer) < 2:
        # User said nothing — just proceed with what we have
        return intent

    # Re-parse with the clarified input
    combined = f"{intent['display_topic']} — {answer}"
    speak("Got it. Let me refine my research plan.")
    return _ai_parse_intent(combined)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Multi-Query Web Search + Scraping
# ══════════════════════════════════════════════════════════════════════════════

def _search_web(query: str) -> list[dict]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=MAX_RESULTS):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href",  ""),
                    "snippet": r.get("body",  "")
                })
    except Exception as e:
        print(f"[research] DDG search error: {e}")
    return results


def _scrape_page(url: str, char_limit: int = 3000) -> str:
    try:
        resp = requests.get(url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; INNOSTAA/5.0)"},
            timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","form","iframe"]):
            tag.decompose()
        lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
        return "\n".join(lines)[:char_limit]
    except Exception:
        return ""


def gather_research_data(queries: list[str], speak) -> tuple[str, list[dict]]:
    """
    Run multiple search queries and merge all results.
    This is what makes the MUN-style research work — we search the right angle.
    """
    speak("Searching the web across multiple angles. Please hold on.")

    all_results, seen_urls = [], set()
    for q in queries:
        print(f"[research] Searching: {q}")
        for r in _search_web(q):
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)

    if not all_results:
        speak("I could not find any results. Please try again.")
        return "", []

    speak(f"Found {len(all_results)} sources across {len(queries)} searches. Extracting content.")
    combined, valid_sources = "", []
    for idx, r in enumerate(all_results, 1):
        text = _scrape_page(r["url"]) or r["snippet"]
        if text:
            combined      += f"\n\n--- Source {idx}: {r['title']} ---\n{text}"
            valid_sources.append(r)
        if len(combined) >= MAX_CHARS:
            break

    return combined[:MAX_CHARS], valid_sources


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — System Prompts per Report Type
# ══════════════════════════════════════════════════════════════════════════════

_SHARED_RULES = """
FORMATTING RULES (follow every rule strictly):

LENGTH: Target 6-10 pages total. Each section: 2-3 tight paragraphs or 6-10 bullets.
Do NOT pad with filler. Every sentence must earn its place.
Do NOT repeat information across sections.

HEADINGS: Use ## for main headings. Choose your OWN headings based on content.
Do NOT use generic headings like Overview, Key Findings, Conclusion.
Use specific, descriptive headings like:
  "Power Rule and Polynomial Derivatives"
  "Economic Benefits of AI for the United States"
  "Key Events 1947-1990"

CAPITALISATION:
- Proper nouns, names, places, laws, theories: always capitalised.
- Acronyms always uppercase: AI, DNA, NASA, CPU, MUN, UN, USA, UK, etc.
- Start every sentence with a capital letter.
- Do NOT title-case every word in body text — only first word of sentences
  and proper nouns. Example: "The power rule states that..." not
  "The Power Rule States That..."

EQUATIONS — ONLY for topics that genuinely involve math/physics/chemistry:
- Tag format: [EQ: formula]  — one formula per tag, on its own line
- Use clean ASCII math: ^ for power, _ for subscript, / for divide
- Examples: [EQ: d/dx(x^n) = n*x^(n-1)]  [EQ: sin^2(x) + cos^2(x) = 1]
- NEVER repeat the formula in plain text near the tag
- For non-math topics: zero equations

BULLETS: Use ONLY "- " prefix. Never use * or numbers unless it is a numbered list.

PLAIN ASCII ONLY: No Unicode symbols. No curly quotes, arrows, degree signs,
subscript numbers. Use: ' for apostrophe, - for dash, -> for arrow.
Write "degrees" not the degree symbol.

Do NOT write source URLs in body text.
Do NOT add preamble — start directly with the first ## heading.
"""

PROMPTS = {

"formula_sheet": """You are INNOSTAA's Research Engine creating a COMPLETE FORMULA SHEET / REFERENCE SHEET.

CRITICAL: The user wants ALL formulas listed — not a prose report. Your job is to list
every single relevant formula, rule, and equation for the topic. This is a reference document,
not an essay. Include EVERY formula mentioned in the sources.

STRUCTURE:
- Organise formulas into logical groups (your choice of group names)
- Each group is a ## heading
- Under each heading: list formulas using [EQ: formula] tags
- After each formula, write ONE short line explaining what it means
- Include ALL formulas from the sources — do not skip any
- A typical formula sheet has 20-50+ formulas — include them all
- Add a "Quick Reference Tips" section at the end with key rules to remember

""" + _SHARED_RULES,


"study_notes": """You are INNOSTAA's Research Engine creating CONCISE STUDY NOTES.

The user wants clear, scannable study notes — key concepts, definitions, important points.
Write in a note-taking style: short sentences, plenty of bullet points, key terms bolded.

STRUCTURE:
- Choose your own headings that match the topic content
- Each section should be scannable in under 30 seconds
- Use bullet points heavily
- Bold all key terms and definitions
- Include any important formulas using [EQ: formula] tags
- End with a "Key Points to Remember" section

""" + _SHARED_RULES,


"timeline": """You are INNOSTAA's Research Engine creating a CHRONOLOGICAL TIMELINE REPORT.

The user wants events in chronological order with context.

STRUCTURE:
- Organise by time periods (your choice of period names as ## headings)
- Under each period: list events as bullet points with years/dates
- Include causes, effects, and significance of major events
- Use specific dates, names, and figures from the sources

""" + _SHARED_RULES,


"analysis": """You are INNOSTAA's Research Engine — a senior analyst producing a DEEP ANALYSIS REPORT.

CRITICAL RULES:
- Use ONLY facts from the web content provided — never invent data
- Write analytically: explain WHY and HOW, not just WHAT
- Each section must have specific data points, examples, and real figures from sources
- No generic filler sentences — every sentence must add specific value

STRUCTURE:
- Choose your OWN headings based on what the content is actually about
- Do NOT use generic headings like Overview or Conclusion
- Use specific headings like "China AI Investment 2024-2035" or "Impact on Manufacturing"
- Minimum 4 paragraphs per section
- End with an "Implications and Outlook" section

""" + _SHARED_RULES,


"speech_debate": """You are INNOSTAA's Research Engine preparing a SPEECH / DEBATE POSITION DOCUMENT.

This is for a MUN speech, debate, or persuasive position paper.
Your FRAMING INSTRUCTIONS will tell you which country/side/angle to write from.
You must argue THAT SPECIFIC POSITION — do not give a neutral overview.

STRUCTURE:
- Opening Statement: a powerful 2-3 sentence opening the speaker can use verbatim
- Then organise arguments into ## headed sections
- Each section = one strong argument with evidence, data, and real-world examples
- Use bullet points for sub-points and supporting evidence
- Include counterargument rebuttals in a dedicated section
- End with: "Closing Lines" — 2-3 punchy sentences to end the speech

CRITICAL:
- Argue the assigned position confidently and persuasively
- Use real data, treaties, policies, and events from the sources
- Every argument needs a supporting fact or statistic
- Write as if the delegate is speaking at the podium

""" + _SHARED_RULES,


"general": """You are INNOSTAA's Research Engine producing a COMPREHENSIVE RESEARCH REPORT.

CRITICAL RULES:
- Use ONLY facts from the web content provided — never invent data
- Write with depth and specificity — every sentence must add value
- Include specific numbers, dates, names where available in sources

STRUCTURE:
- Choose your OWN headings that match the actual content
- Do NOT use generic headings like Overview, Key Findings, Conclusion
- Use descriptive headings that tell the reader exactly what that section covers
- Minimum 3 paragraphs per section

""" + _SHARED_RULES,

}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — AI Report Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(display_topic: str, research_data: str,
                    framing: str, report_type: str, speak) -> str:
    speak("Analysing content and generating your report with A I.")

    system_prompt = PROMPTS.get(report_type, PROMPTS["general"])

    # Inject framing context directly into the user message
    framing_line = f"\nFRAMING INSTRUCTIONS: {framing}" if framing.strip() else ""

    user_msg = (
        f"Research Topic: {display_topic}\n"
        f"{framing_line}\n\n"
        f"Web Content (use ONLY facts from here — include EVERY relevant "
        f"formula/fact/data point/argument from the sources):\n{research_data}\n\n"
        f"Write the complete {report_type.replace('_',' ')} now. "
        f"Be exhaustive — do not skip formulas, arguments, or important facts. "
        f"Use your own descriptive headings. "
        f"All equations must use [EQ: formula] tags only."
    )

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=2000   # was 4096 — keeps report to ~6-10 pages instead of 24
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        speak("There was an error generating the report.")
        print(f"[research] Groq error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Filename Generator
# ══════════════════════════════════════════════════════════════════════════════

def _generate_filename(display_topic: str, report_type: str) -> str:
    type_suffix = {
        "formula_sheet": "Formulas",
        "study_notes":   "Notes",
        "timeline":      "Timeline",
        "analysis":      "Analysis",
        "speech_debate": "Speech",
        "general":       "Report"
    }.get(report_type, "Report")

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"Give a very short 1-3 word TitleCase NoSpaces slug for: '{display_topic}'. "
                f"Examples: Derivatives, ChinaAI, ClimateChange, BlackHoles, USAonAI, MUNDebate. "
                f"Return ONLY the slug, nothing else."}],
            temperature=0.1,
            max_tokens=12
        )
        slug = resp.choices[0].message.content.strip().strip("'\"` \n")
        slug = re.sub(r"[^\w]", "", slug)[:20]
    except Exception:
        words = [w for w in display_topic.split()
                 if w.lower() not in {"of","on","in","at","to","a","an","the","and","for"}]
        slug  = "".join(w.capitalize() for w in words[:3])
        slug  = re.sub(r"[^\w]", "", slug)[:20]

    date_part = datetime.now().strftime("%b%d")
    return f"{slug}_{type_suffix}_{date_part}.pdf"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Unicode Sanitiser
# ══════════════════════════════════════════════════════════════════════════════

_UNICODE_MAP = {
    "\u2018":"'",  "\u2019":"'",  "\u201c":'"',  "\u201d":'"',
    "\u2032":"'",  "\u2033":'"',
    "\u2013":"-",  "\u2014":"-",  "\u2212":"-",
    "\u2022":"-",  "\u2192":"->", "\u2190":"<-", "\u2194":"<->",
    "\u2191":"^",  "\u2193":"v",
    "\u00b2":"^2", "\u00b3":"^3", "\u00b9":"^1",
    "\u2070":"^0", "\u2074":"^4", "\u2075":"^5",
    "\u2076":"^6", "\u2077":"^7", "\u2078":"^8", "\u2079":"^9",
    "\u2080":"_0", "\u2081":"_1", "\u2082":"_2", "\u2083":"_3",
    "\u2084":"_4", "\u2085":"_5", "\u2086":"_6", "\u2087":"_7",
    "\u2088":"_8", "\u2089":"_9",
    "\u00b0":" degrees", "\u00b1":"+/-", "\u00d7":"x", "\u00f7":"/",
    "\u2248":"~",  "\u2260":"!=", "\u2264":"<=", "\u2265":">=",
    "\u221e":"infinity", "\u221a":"sqrt",
    "\u03c0":"pi", "\u03b1":"alpha", "\u03b2":"beta", "\u03b3":"gamma",
    "\u03b4":"delta", "\u03bb":"lambda", "\u03bc":"mu",
    "\u03c3":"sigma", "\u03c9":"omega", "\u03b8":"theta",
    "\u03c6":"phi", "\u03c8":"psi",
    "\u20b9":"Rs.", "\u20ac":"EUR", "\u00a3":"GBP", "\u00a5":"JPY",
    "\u00a9":"(c)", "\u00ae":"(R)", "\u2122":"(TM)",
    "\u2026":"...", "\u00b7":".", "\u00a0":" ",
    "\u2243":"~=", "\u2261":"===",
    "\u222b":"integral", "\u2211":"sum", "\u220f":"product",
    "\u2202":"d",  "\u2207":"grad",
    "\u221d":"proportional to",
}


def _sanitise(text: str) -> str:
    for char, rep in _UNICODE_MAP.items():
        text = text.replace(char, rep)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text


def _xml(text: str) -> str:
    """
    Escape text for safe use inside ReportLab Paragraph XML.
    Order matters: & must be escaped FIRST.
    We escape ALL < > then selectively restore only the tags
    _process_inline intentionally produces (<b>, </b>, <i>, </i>, <br/>).
    This prevents any stray LLM arrow like -> or <-> or <<quote>> from
    breaking the ReportLab XML parser and corrupting the page into binary.
    """
    text = _sanitise(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # Restore only the tags we deliberately emit
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    text = text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    text = text.replace("&lt;br/&gt;", "<br/>")
    return text


def _safe_paragraph(text: str, style) -> "Paragraph":
    """
    Wrap Paragraph() so a single bad line never crashes the whole PDF build.
    Falls back to stripped plain text on any XML parse error.
    """
    try:
        return Paragraph(text, style)
    except Exception as e:
        print(f"[research] Paragraph XML error, falling back to plain: {e}")
        plain = re.sub(r'<[^>]+>', '', text)
        plain = plain.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        plain = _sanitise(plain)
        try:
            return Paragraph(plain, style)
        except Exception:
            return Paragraph("(content rendering error)", style)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Capitalisation Post-Processing
# ══════════════════════════════════════════════════════════════════════════════

_CAP_FIXES = {
    r"\bnasa\b":"NASA", r"\bdna\b":"DNA",   r"\brna\b":"RNA",
    r"\busa\b":"USA",   r"\buk\b":"UK",     r"\beu\b":"EU",
    r"\bun\b":"UN",     r"\bcpu\b":"CPU",   r"\bgpu\b":"GPU",
    r"\bram\b":"RAM",   r"\bapi\b":"API",   r"\biot\b":"IoT",
    r"\bai\b":"AI",     r"\bml\b":"ML",     r"\bwifi\b":"WiFi",
    r"\bstem\b":"STEM", r"\blhc\b":"LHC",   r"\bcern\b":"CERN",
    r"\bwho\b":"WHO",   r"\bnato\b":"NATO", r"\bgpt\b":"GPT",
    r"\bllm\b":"LLM",   r"\bphd\b":"PhD",   r"\bccp\b":"CCP",
    r"\bmun\b":"MUN",
}


def _fix_caps(text: str) -> str:
    eq_tags = re.findall(r'\[EQ:[^\]]+\]', text)
    for i, tag in enumerate(eq_tags):
        text = text.replace(tag, f"__EQ_{i}__", 1)

    for pattern, rep in _CAP_FIXES.items():
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)

    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

    for i, tag in enumerate(eq_tags):
        text = text.replace(f"__EQ_{i}__", tag, 1)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — Equation Renderer
# ══════════════════════════════════════════════════════════════════════════════

def _render_equation(eq_text: str) -> str | None:
    try:
        raw = eq_text.strip()
        raw = re.sub(r'd/dx\((.+?)\)', r'\\frac{d}{dx}(\1)', raw)
        raw = re.sub(r'd/dx',          r'\\frac{d}{dx}',     raw)
        raw = re.sub(r'(\w+)\s*/\s*(\w+)', r'\\frac{\1}{\2}', raw)
        raw = re.sub(r'\^([A-Za-z0-9]+(?:\([^)]+\))?)',
                     lambda m: f'^{{{m.group(1)}}}', raw)
        raw = re.sub(r'_([A-Za-z0-9]+)',
                     lambda m: f'_{{{m.group(1)}}}', raw)
        replacements = [
            ("sqrt(",   r"\sqrt{"),  ("sin(",  r"\sin("),
            ("cos(",    r"\cos("),   ("tan(",  r"\tan("),
            ("log(",    r"\log("),   ("ln(",   r"\ln("),
            ("lim",     r"\lim"),    ("->",    r"\to "),
            ("+-",      r"\pm "),    ("<=",    r"\leq "),
            (">=",      r"\geq "),   ("!=",    r"\neq "),
            ("*",       r"\cdot "),  ("inf",   r"\infty"),
            ("alpha",   r"\alpha"),  ("beta",  r"\beta"),
            ("theta",   r"\theta"),  ("pi",    r"\pi"),
            ("lambda",  r"\lambda"), ("sigma", r"\sigma"),
            ("integral",r"\int"),    ("sum",   r"\sum"),
        ]
        for plain, latex in replacements:
            raw = raw.replace(plain, latex)

        math_str = f"${raw}$"

        fig, ax = plt.subplots(figsize=(6, 0.8))
        ax.axis("off")
        fig.patch.set_facecolor("#EEF0FF")
        ax.set_facecolor("#EEF0FF")
        ax.text(0.5, 0.5, math_str,
                fontsize=14, ha="center", va="center",
                transform=ax.transAxes,
                color="#1A1A2E",
                usetex=False)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight",
                    facecolor="#EEF0FF", edgecolor="none", pad_inches=0.1)
        plt.close(fig)
        return tmp.name
    except Exception as e:
        print(f"[research] Equation render error '{eq_text}': {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — Image Handling
# ══════════════════════════════════════════════════════════════════════════════

_VISUAL_TOPICS = {
    "physics","space","astronomy","biology","chemistry","geology",
    "geography","nature","wildlife","animals","plants","earth",
    "ocean","climate","weather","architecture","engineering",
    "medicine","anatomy","robotics","manufacturing","solar","planet",
    "galaxy","black hole","dna","cell","quantum","nuclear","energy",
    "electricity","environment","maps","landmarks","history","art",
}

_TEXT_ONLY_TOPICS = {
    "formula","formulas","equation","equations","derivative","derivatives",
    "integral","integrals","theorem","theorems","mun","policy","politics",
    "law","economics","finance","banking","governance","diplomacy",
    "philosophy","ethics","sociology","psychology","literature","religion",
    "business","management","strategy","notes","class 12","class 11",
    "class 10","cbse","ncert","syllabus","chapter","speech","debate",
    "position paper","resolution","argument","arguments",
}


def _topic_needs_images(query: str, report_type: str) -> bool:
    if report_type in ("formula_sheet", "study_notes", "speech_debate"):
        return False
    q = query.lower()
    for phrase in _TEXT_ONLY_TOPICS:
        if phrase in q:
            return False
    for phrase in _VISUAL_TOPICS:
        if phrase in q:
            return True
    return False


def _is_text_heavy(pil_img) -> bool:
    try:
        small  = pil_img.resize((100, 100)).convert("RGB")
        pixels = list(small.getdata())
        light  = sum(1 for r, g, b in pixels if r > 220 and g > 220 and b > 220)
        return (light / len(pixels)) > 0.72
    except Exception:
        return False


def _get_best_image(section_heading: str, query: str) -> str | None:
    try:
        urls = []
        with DDGS() as ddgs:
            for r in ddgs.images(f"{query} {section_heading}", max_results=6):
                url = r.get("image", "")
                if url and url.startswith("http"):
                    urls.append(url)
    except Exception:
        return None

    best_bytes, best_px = None, 0
    for url in urls[:5]:
        try:
            resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"},
                                timeout=7, stream=True)
            if resp.status_code != 200:
                continue
            pil = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
            if pil.width < 250 or pil.height < 180:
                continue
            if _is_text_heavy(pil):
                continue
            px = pil.width * pil.height
            if px > best_px:
                best_px    = px
                if pil.width > 800:
                    pil = pil.resize((800, int(pil.height * 800 / pil.width)),
                                     PILImage.LANCZOS)
                buf = io.BytesIO()
                pil.save(buf, "JPEG", quality=85)
                best_bytes = buf.getvalue()
        except Exception:
            continue

    if best_bytes:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        PILImage.open(io.BytesIO(best_bytes)).save(tmp.name, "JPEG", quality=88)
        return tmp.name
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — PDF Styles
# ══════════════════════════════════════════════════════════════════════════════

def _build_styles() -> dict:
    s = {}
    s["cover_title"] = ParagraphStyle(
        "cover_title", fontName="Helvetica-Bold", fontSize=32,
        textColor=C_WHITE, alignment=TA_CENTER, leading=38)
    s["cover_sub"] = ParagraphStyle(
        "cover_sub", fontName="Helvetica", fontSize=15,
        textColor=colors.HexColor("#E0E0E0"), alignment=TA_CENTER, leading=20)
    s["cover_type"] = ParagraphStyle(
        "cover_type", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_ACCENT, alignment=TA_CENTER, leading=16)
    s["cover_meta"] = ParagraphStyle(
        "cover_meta", fontName="Helvetica-Oblique", fontSize=10,
        textColor=colors.HexColor("#AAAAAA"), alignment=TA_CENTER, leading=14)
    s["section_heading"] = ParagraphStyle(
        "section_heading", fontName="Helvetica-Bold", fontSize=13,
        textColor=C_WHITE, alignment=TA_LEFT, leading=17)
    s["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10.5,
        textColor=C_TEXT, alignment=TA_JUSTIFY, leading=17,
        spaceBefore=3, spaceAfter=3)
    s["formula_label"] = ParagraphStyle(
        "formula_label", fontName="Helvetica", fontSize=10,
        textColor=colors.HexColor("#444466"), alignment=TA_LEFT,
        leading=14, spaceBefore=1, spaceAfter=5, leftIndent=8)
    s["bullet"] = ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=10.5,
        textColor=C_TEXT, alignment=TA_LEFT, leading=16,
        spaceBefore=2, spaceAfter=2, leftIndent=16, firstLineIndent=-10)
    s["img_caption"] = ParagraphStyle(
        "img_caption", fontName="Helvetica-Oblique", fontSize=8.5,
        textColor=C_SUBTLE, alignment=TA_CENTER, leading=12, spaceAfter=5)
    s["sources_heading"] = ParagraphStyle(
        "sources_heading", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_ACCENT, alignment=TA_LEFT, leading=14,
        spaceBefore=14, spaceAfter=6)
    s["source_item"] = ParagraphStyle(
        "source_item", fontName="Helvetica", fontSize=9,
        textColor=colors.HexColor("#333366"), alignment=TA_LEFT,
        leading=13, spaceAfter=3, leftIndent=12)
    s["footer"] = ParagraphStyle(
        "footer", fontName="Helvetica-Oblique", fontSize=8,
        textColor=C_SUBTLE, alignment=TA_CENTER)
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — PDF Building Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dark_table(para, top_pad=8, bot_pad=8):
    t = Table([[para]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_DARK),
        ("TOPPADDING",    (0,0),(-1,-1), top_pad),
        ("BOTTOMPADDING", (0,0),(-1,-1), bot_pad),
        ("LEFTPADDING",   (0,0),(-1,-1), 16),
        ("RIGHTPADDING",  (0,0),(-1,-1), 16),
    ]))
    return t


_TYPE_LABELS = {
    "formula_sheet": "Formula Sheet",
    "study_notes":   "Study Notes",
    "timeline":      "Timeline",
    "analysis":      "Analysis Report",
    "speech_debate": "Speech and Debate Prep",   # & → and  (avoids XML crash in cover)
    "general":       "Research Report",
}


def _cover_page(styles, display_topic, report_type, timestamp) -> list:
    label = _TYPE_LABELS.get(report_type, "Research Report")
    fl = [Spacer(1, 34*mm)]
    fl.append(_dark_table(_safe_paragraph("INNOSTAA", styles["cover_title"]),                       top_pad=22, bot_pad=4))
    fl.append(_dark_table(_safe_paragraph(label,       styles["cover_type"]),                        top_pad=4,  bot_pad=4))
    fl.append(_dark_table(_safe_paragraph(f"Topic: {_xml(display_topic)}", styles["cover_sub"]),     top_pad=4,  bot_pad=4))
    fl.append(_dark_table(_safe_paragraph(f"Generated on {timestamp}", styles["cover_meta"]),        top_pad=6,  bot_pad=6))
    fl.append(_dark_table(_safe_paragraph(
        "Powered by Groq llama-3.3-70b-versatile and DuckDuckGo",
        styles["cover_meta"]), top_pad=4, bot_pad=22))
    fl += [Spacer(1, 10*mm),
           HRFlowable(width="100%", thickness=2, color=C_ACCENT, spaceAfter=4),
           PageBreak()]
    return fl


def _section_heading_block(heading, styles) -> list:
    ht = Table([[_safe_paragraph(_xml(heading), styles["section_heading"])]], colWidths=[PAGE_W])
    ht.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_MID),
        ("TOPPADDING",    (0,0),(-1,-1), 9),
        ("BOTTOMPADDING", (0,0),(-1,-1), 9),
        ("LEFTPADDING",   (0,0),(-1,-1), 14),
        ("RIGHTPADDING",  (0,0),(-1,-1), 14),
    ]))
    return [ht, Spacer(1, 3*mm)]


def _embed_equation(eq_text: str, styles: dict) -> list:
    fl   = [Spacer(1, 3*mm)]
    path = _render_equation(eq_text)
    if path and os.path.exists(path):
        try:
            pil        = PILImage.open(path)
            w_px, h_px = pil.size
            desired_w  = min(w_px * 0.65, 220)
            scale      = desired_w / w_px
            desired_h  = h_px * scale

            img = RLImage(path, width=desired_w, height=desired_h)
            t   = Table([[img]], colWidths=[PAGE_W])
            t.setStyle(TableStyle([
                ("ALIGN",         (0,0),(-1,-1), "CENTER"),
                ("BACKGROUND",    (0,0),(-1,-1), C_EQ_BG),
                ("TOPPADDING",    (0,0),(-1,-1), 8),
                ("BOTTOMPADDING", (0,0),(-1,-1), 8),
            ]))
            fl.append(t)
        except Exception as e:
            print(f"[research] Equation embed error: {e}")
            fl.append(_safe_paragraph(f"[ {_xml(eq_text)} ]", styles["body"]))
    else:
        fl.append(_safe_paragraph(f"[ {_xml(eq_text)} ]", styles["body"]))

    fl.append(Spacer(1, 2*mm))
    return fl


def _embed_image(img_path, caption, styles) -> list:
    fl = []
    if not img_path or not os.path.exists(img_path):
        return fl
    try:
        pil        = PILImage.open(img_path)
        w_px, h_px = pil.size
        scale      = min(IMG_MAX_W / (w_px * 0.75),
                         IMG_MAX_H / (h_px * 0.75), 1.0)
        fl.append(Spacer(1, 4*mm))
        rl_img = RLImage(img_path, width=w_px*0.75*scale, height=h_px*0.75*scale)
        t = Table([[rl_img]], colWidths=[PAGE_W])
        t.setStyle(TableStyle([
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("BOX",           (0,0),(-1,-1), 0.5, C_IMG_BDR),
        ]))
        fl.append(t)
        if caption:
            fl.append(_safe_paragraph(_xml(caption), styles["img_caption"]))
        fl.append(Spacer(1, 4*mm))
    except Exception as e:
        print(f"[research] Image embed error: {e}")
    return fl


def _process_inline(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\[EQ:[^\]]+\]', '', text)
    return text.strip()


def _render_body(body: str, styles: dict, report_type: str) -> list:
    fl     = []
    lines  = body.splitlines()
    i      = 0

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            fl.append(Spacer(1, 2*mm))
            i += 1
            continue

        eq_matches = re.findall(r'\[EQ:\s*([^\]]+)\]', line)
        if eq_matches:
            pre = re.sub(r'\[EQ:[^\]]+\]', '', line).strip()
            if pre:
                fl.append(_safe_paragraph(_process_inline(_xml(pre)), styles["body"]))
            for eq in eq_matches:
                fl += _embed_equation(eq.strip(), styles)
                if report_type == "formula_sheet":
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        label_line = lines[j].strip()
                        if not label_line.startswith("[EQ:") and not label_line.startswith("##") and not label_line.startswith("-"):
                            fl.append(_safe_paragraph(_xml(label_line), styles["formula_label"]))
                            i = j
            i += 1
            continue

        if line.strip().startswith(("- ", "* ")):
            content = line.strip()[2:].strip()
            fl.append(_safe_paragraph(
                "&#8226;  " + _process_inline(_xml(content)),
                styles["bullet"]))
            i += 1
            continue

        fl.append(_safe_paragraph(_process_inline(_xml(line.strip())), styles["body"]))
        i += 1

    fl.append(Spacer(1, 2*mm))
    return fl


def _parse_sections(report_text: str) -> list[tuple[str, str]]:
    sections, cur_h, cur_b = [], None, []
    for line in report_text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            if cur_h is not None:
                sections.append((cur_h, "\n".join(cur_b).strip()))
            cur_h, cur_b = s[3:].strip(), []
        elif cur_h is not None:
            cur_b.append(line)
    if cur_h:
        sections.append((cur_h, "\n".join(cur_b).strip()))
    return sections


def _sources_block(sources, styles) -> list:
    fl = [
        HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceBefore=4, spaceAfter=4),
        _safe_paragraph("References and Sources", styles["sources_heading"])
    ]
    for idx, s in enumerate(sources, 1):
        fl.append(_safe_paragraph(
            f"[{idx}]&nbsp; <b>{_xml(s.get('title','Untitled'))}</b>"
            f"<br/>{_xml(s.get('url',''))}",
            styles["source_item"]))
    return fl


def _footer_block(styles) -> list:
    return [
        Spacer(1, 8*mm),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC")),
        _safe_paragraph(
            "This report was automatically generated by INNOSTAA. For informational purposes only.",
            styles["footer"])
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — PDF Assembly
# ══════════════════════════════════════════════════════════════════════════════

def save_pdf(display_topic, report_text, sources, report_type, speak) -> str:
    speak("Building your P D F report.")

    filename  = _generate_filename(display_topic, report_type)
    filepath  = os.path.join(DOWNLOADS_DIR, filename)
    styles    = _build_styles()
    story     = []
    timestamp = datetime.now().strftime("%d %B %Y, %I:%M %p")

    story += _cover_page(styles, display_topic, report_type, timestamp)

    sections   = _parse_sections(report_text)
    use_images = _topic_needs_images(display_topic, report_type)

    if not sections:
        story.append(_safe_paragraph(_xml(report_text), styles["body"]))
    else:
        if use_images:
            speak("Fetching relevant images for sections.")
        for heading, body in sections:
            story += _section_heading_block(heading, styles)
            story += _render_body(body, styles, report_type)
            if use_images:
                img_path = _get_best_image(heading, display_topic)
                if img_path:
                    story += _embed_image(img_path, f"{heading} — {display_topic}", styles)
            story.append(Spacer(1, 5*mm))

    if sources:
        story += _sources_block(sources, styles)
    story += _footer_block(styles)

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        title=f"INNOSTAA – {display_topic}",
        author="INNOSTAA Desktop Assistant"
    )
    doc.build(story)
    return filepath


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — Public Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _strip_trigger_phrases(text: str) -> str:
    """
    Remove ONLY the activation trigger from the front of the user's sentence
    so the AI parser receives a clean topic/request, not the launch command.

    Examples:
      "research report"           → ""          (bare trigger, no topic)
      "make a research report"    → ""          (bare trigger, no topic)
      "research on climate change"→ "climate change"
      "make a report on MUN AI"   → "MUN AI"
      "study notes for WW2"       → "study notes for WW2"  (left alone — has real content)
    """
    t = text.strip()
    lower = t.lower()

    # Phrases that are PURE triggers with no topic attached — wipe them entirely
    bare_triggers = [
        "research report", "make a research report", "make research report",
        "start research", "research mode", "do research", "make a report",
        "generate report", "generate a report", "create a report",
        "create report", "write a report", "write report",
    ]
    for trigger in bare_triggers:
        if lower == trigger or lower == trigger + ".":
            return ""   # Nothing left — we must ask for the topic

    # Phrases at the START that are just the verb+object launcher, strip them
    # but only if something meaningful follows
    prefix_triggers = [
        "research on ", "research about ", "research ",
        "make a report on ", "make a report about ",
        "generate a report on ", "generate a report about ",
        "create a report on ", "create a report about ",
        "write a report on ", "write a report about ",
        "make report on ", "make report about ",
        "prepare a report on ", "prepare report on ",
    ]
    for prefix in prefix_triggers:
        if lower.startswith(prefix):
            remainder = t[len(prefix):].strip()
            if len(remainder) >= 3:
                return remainder   # Return what comes AFTER the trigger
            else:
                return ""          # Nothing meaningful after trigger

    # No trigger found — pass the full text through unchanged
    return t

def _history_context(n=10):
    try:
        ctx = history.context(n=n)
        return "\n".join(
            f"{m['role']}: {m['content']}"
            for m in ctx
        )
    except Exception:
        return "" 
    
def start(user_text: str, speak, listen):
    """
    Main entry point — called from INNOSTAA's process() function.

        import research
        if 'research' in intent:
            research.start(user_text, speak=speak, listen=listen)
            return

    v5.2 additions:
      - Cancel at any point by saying stop / cancel / never mind / etc.
      - _CANCELLED flag lets external code (e.g. a voice command) abort too.
      - Cancel is checked: before parsing, after every listen(), before search,
        before generate, before PDF — so it stops at the earliest opportunity.
      - After completing ONE report, start() returns immediately — no loop.
    """
    global _CANCELLED
    _CANCELLED = False   # reset from any previous session

    text = _strip_trigger_phrases(user_text.strip())

    # ── If bare trigger, ask for topic — but honour cancel ────────────────────
    if len(text) < 3:
        speak("What topic would you like me to research? Say stop anytime to cancel.")
        text = (listen() or "").strip()
        if not text or _is_cancel(text):
            speak("No problem. Research cancelled.")
            return
        if len(text) < 3:
            speak("I didn't catch that. Cancelling research.")
            return

    # ── Cancel check after we have the initial text ───────────────────────────
    if _is_cancel(text) or _check_cancel(speak):
        return

    # Step 1: AI understands intent
    speak("Let me understand what you need.")
    intent = _ai_parse_intent(text)

    if _check_cancel(speak):
        return

    # Step 2: Clarifying question (only if AI flagged it)
    if intent.get("needs_clarify"):
        q = intent.get("clarify_question", "Could you tell me more?")
        speak(q)
        answer = (listen() or "").strip()

        if not answer or _is_cancel(answer) or _check_cancel(speak):
            speak("Research cancelled. No problem.")
            return

        combined = f"{intent['display_topic']} — {answer}"
        speak("Got it. Let me refine my research plan.")
        intent = _ai_parse_intent(combined)

        if _check_cancel(speak):
            return

    display_topic  = intent["display_topic"]
    search_queries = intent["search_queries"]
    report_type    = intent["report_type"]
    framing        = intent["framing"]

    # ── Guard: vague topic — ask once more ───────────────────────────────────
    if not search_queries or display_topic.lower() in (
            "unknown topic", "general report", "unknown", "none", "n/a", ""):
        speak("I couldn't figure out the exact topic. What would you like me to research?")
        clarified = (listen() or "").strip()

        if not clarified or _is_cancel(clarified) or _check_cancel(speak):
            speak("Research cancelled.")
            return

        intent         = _ai_parse_intent(clarified)
        display_topic  = intent["display_topic"]
        search_queries = intent["search_queries"]
        report_type    = intent["report_type"]
        framing        = intent["framing"]

    if _check_cancel(speak):
        return

    # ── Confirm before starting the long search ───────────────────────────────
    label = _TYPE_LABELS.get(report_type, "report")
    speak(
        f"I understood that you want a {label} on {display_topic}. "
        f"Should I start the research?"
    )
    
    confirmation = (listen() or "").strip().lower()
    
    if _is_cancel(confirmation):
        speak("Research cancelled.")
        return
    
    positive = {
        "yes",
        "yeah",
        "yep",
        "sure",
        "go ahead",
        "start",
        "continue",
        "okay",
        "ok"
    }
    
    if not any(word in confirmation for word in positive):
        speak("Okay, research cancelled.")
        return
    research_data, sources = gather_research_data(search_queries, speak)
    if not research_data:
        return

    if _check_cancel(speak):
        return

    # Step 4: Generate
    report_text = generate_report(display_topic, research_data, framing, report_type, speak)
    if not report_text:
        return

    if _check_cancel(speak):
        return

    # Step 5: Post-process
    report_text = _fix_caps(report_text)
    report_text = _sanitise(report_text)

    if _check_cancel(speak):
        return

    # Step 6: Build PDF
    try:
        filepath = save_pdf(display_topic, report_text, sources, report_type, speak)
        filename = os.path.basename(filepath)
        speak(
            f"Done! Your {label} on {display_topic} "
            f"has been saved to Downloads as {filename}."
        )
    except Exception as e:
        print(f"[research] PDF error: {e}")
        import traceback; traceback.print_exc()
        speak("The report was generated but I had trouble saving the PDF. "
              "Please check the console.")
    # start() ends here — intentionally no loop, no follow-up listen