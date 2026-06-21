"""
latest_data.py  –  Web-search-only module for INNOSTAA
-------------------------------------------------------
Rules:
  • Answers come ONLY from DDGS web search snippets.
  • No LLM training data is ever used to answer questions.
  • Training-data answers live in ai_reply() in innostaa_pyttsx3.py.

Flow
────
  1. Real date/time injected into every AI call so queries are always current.
  2. context_store (city, user prefs) injected into every decision so the AI
     enriches queries automatically (e.g. "weather" → "weather Ballabgarh 2026").
  3. AI decides: search now, or ask a follow-up first.
     Follow-ups are spoken directly as a plain string — no sentinel needed.
     The pending query is stored internally; next fetch() resolves it.
  4. DDGS search → if snippets answer the question, summarise.
     If snippets are weak, automatically retry with a rephrased query.
  5. If both attempts fail, say so honestly. No hallucination.

Public API
──────────
  fetch(query) → str          (always a speakable string now)
  context_store.get(key) / .set(key, value)
"""

from __future__ import annotations

import os
import re
import json
from datetime import datetime
import dotenv
dotenv.load_dotenv()

import requests
from ddgs import DDGS
from conversation_history import history, SRC_SEARCH

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── Context Store ─────────────────────────────────────────────────────────────
_CONTEXT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "automation_data", "search_context.json"
)
os.makedirs(os.path.dirname(_CONTEXT_FILE), exist_ok=True)


class _ContextStore:
    def __init__(self):
        self._data: dict = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(_CONTEXT_FILE):
                with open(_CONTEXT_FILE, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception as e:
            print(f"[context_store] Load error: {e}")
            self._data = {}

    def _save(self):
        try:
            with open(_CONTEXT_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"[context_store] Save error: {e}")

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: str):
        if not key or not value:
            return
        value = value.strip()
        if self._data.get(key) == value:
            return
        self._data[key] = value
        self._save()
        print(f"[context_store] Saved {key!r} = {value!r}")

    def as_prompt_line(self) -> str:
        """Return a single line suitable for injecting into a system prompt."""
        if not self._data:
            return ""
        parts = []
        if self._data.get("city"):
            parts.append(f"user's city: {self._data['city']}")
        for k, v in self._data.items():
            if k != "city":
                parts.append(f"{k}: {v}")
        return "Known user context — " + ", ".join(parts) + "."

    def summary(self) -> str:
        if not self._data:
            return "none"
        return "; ".join(f"{k}: {v}" for k, v in self._data.items())


context_store = _ContextStore()

# ── Pending follow-up state (module-level, thread-safe enough for single user) -
_pending_query: str = ""   # the query that triggered a follow-up


# ── Low-level Groq helper ─────────────────────────────────────────────────────

def _groq(messages: list[dict], max_tokens: int = 300) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "temperature": 0.3,
        "max_tokens":  max_tokens,
    }
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        print(f"[latest_data] Groq error: {e}")
        return ""


def _clean(text: str) -> str:
    """Strip markdown so TTS reads cleanly."""
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\n{2,}", ". ", text)
    return text.replace("\n", " ").strip()


# ── DDGS web search ───────────────────────────────────────────────────────────

_SNIPPET_CHAR_LIMIT = 2000   # more room = better answers


def _web_search(query: str, is_news: bool = False) -> str:
    """Search DuckDuckGo. Returns snippet string or '' on total failure."""

    def _run(timelimit=None):
        with DDGS() as ddgs:
            kwargs = dict(region="in-en", max_results=8)
            if timelimit:
                kwargs["timelimit"] = timelimit
            if is_news:
                return list(ddgs.news(query, **kwargs))
            return list(ddgs.text(query, **kwargs))

    raw = []
    try:
        raw = _run(timelimit="m")
        if not raw:
            raw = _run()          # retry without time filter
    except Exception as e:
        print(f"[latest_data] DDGS error: {e}")
        try:
            raw = _run()
        except Exception as e2:
            print(f"[latest_data] DDGS retry failed: {e2}")
            return ""

    if not raw:
        return ""

    parts, total = [], 0
    for item in raw:
        title = item.get("title", "").strip()
        body  = item.get("body", item.get("snippet", "")).strip()
        chunk = f"[{title}] {body}"
        if total + len(chunk) > _SNIPPET_CHAR_LIMIT:
            remaining = _SNIPPET_CHAR_LIMIT - total
            if remaining > 80:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)

    result = "\n".join(parts)
    print(f"[latest_data] DDGS → {len(raw)} results, {len(result)} chars")
    return result


# ── Decision system prompt (built fresh each call with live date + context) ───

def _build_decision_prompt(now: datetime) -> str:
    date_line    = now.strftime("Today is %A, %d %B %Y. Current time: %I:%M %p.")
    context_line = context_store.as_prompt_line()

    return f"""\
You are the search query planner for INNOSTAA, a voice assistant.
{date_line}
{context_line}

Your job: given the user's message and recent conversation, decide whether to
search the web immediately, or ask ONE follow-up question first.

=== WHEN TO FOLLOW UP ===
Ask a follow-up ONLY when the query is genuinely ambiguous and you cannot build
a useful search query even with the conversation history.
Example: user says "tell me about it" with no prior context → ask what topic.
Example: user says "search about the match" with no sport/team mentioned → ask which match.

=== WHEN TO SEARCH IMMEDIATELY ===
Search for everything else. Use the known context (city, year, etc.) to make
the query specific. Always append the current year to time-sensitive queries.
=== think like ===
when you get any query don't summarise to search on internet, while think like if you were be at users place, giving you some example, it shows how to think an ddon't be dependent on the example, it just show how to think, user can ask anything not the thing that is written in the example:
ex 1: user is asking to search fot he points to speak in MUN. now, don't directly search for the points on internet, you can search for topic and then separate the points the search results like which points can be spoke in MUN, like which are latest, which can be more relevent, which can be more effective.
you can alos ask for follow up. 

Examples of good search queries:
  user: "weather"           → "current weather Ballabgarh 2026"  (using saved city)
  user: "who is the PM"     → "Prime Minister of India May 2026"
  user: "latest AI news"    → "artificial intelligence news May 2026"
  user: "search about that" (prior topic: quantum computing) → "quantum computing latest 2026"
note- for weather use date and year both.
=== OUTPUT FORMAT ===
Reply ONLY with valid JSON — no prose, no markdown fences:
{{
  "action":       "search" | "followup",
  "search_query": "<specific web-search query — required when action=search>",
  "question":     "<one spoken follow-up question — required when action=followup>",
  "reasoning":    "<one sentence>"
}}"""


# ── Rephrase prompt (used on retry when first snippets are weak) ──────────────

def _rephrase_query(original_query: str, original_search: str, now: datetime) -> str:
    """Ask the AI to produce an alternative search query."""
    year = now.year
    messages = [
        {
            "role": "system",
            "content": (
                f"Today is {now.strftime('%A, %d %B %Y')}. "
                "You are a search query optimizer. The first search query did not return "
                "a useful answer. Produce ONE alternative search query that is more specific, "
                "uses different keywords, or adds context like the year or location. "
                "Reply with ONLY the query string — no explanation, no quotes."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User's original question: {original_query}\n"
                f"First search query that failed: {original_search}\n"
                f"Current year: {year}\n"
                "Alternative query:"
            ),
        },
    ]
    result = _groq(messages, max_tokens=60).strip().strip('"').strip("'")
    print(f"[latest_data] Rephrased query: {result!r}")
    return result or original_query


# ── Summariser ────────────────────────────────────────────────────────────────

def _summarise(question: str, snippets: str, now: datetime) -> str:
    """
    Summarise DDGS snippets into a spoken answer.
    The AI knows the current date so it can correctly interpret snippet dates.
    Never uses training data — only what the snippets contain.
    """
    date_line = now.strftime("Today is %A, %d %B %Y.")
    system = (
        f"{date_line} "
        "You are summarising live web search results for INNOSTAA, a voice assistant. "
        "Give the answer in 2 to 4 short natural spoken sentences. "
        "RULES: "
        "1. Extract and state facts directly from the search results. "
        "   If the answer is clearly in the snippets, say it confidently. "
        "2. Do NOT use your own training data. Do NOT guess. "
        "3. No bullet points, no markdown, no URLs. "
        "4. If multiple snippets agree on a fact, that fact is reliable — state it. "
        "5. Only if the snippets contain absolutely no relevant information at all, "
        "   say: I could not find a clear answer for that."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Question: {question}\n\nSearch results:\n{snippets}"},
    ]
    return _groq(messages, max_tokens=320)


def _is_weak_answer(text: str) -> bool:
    """Return True if the summariser admitted it couldn't answer."""
    weak_phrases = [
        "could not find",
        "did not clearly answer",
        "no relevant",
        "not find a clear",
        "unable to find",
        "no information",
        "don't have",
        "cannot find",
    ]
    lower = text.lower()
    return any(p in lower for p in weak_phrases)


# ── City extraction ───────────────────────────────────────────────────────────

_LOCATION_RE = re.compile(
    r"\b(weather|temperature|forecast|rain|humidity|wind|aqi|air quality"
    r"|pollution|sunrise|sunset)\b",
    re.IGNORECASE,
)

_NOT_A_CITY = {
    "me", "us", "now", "today", "tomorrow", "yesterday", "the", "a", "an",
    "my", "your", "here", "there", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "neet", "jee", "upsc",
    "india", "after", "before", "about", "leak", "exam", "date", "result",
    "news", "time", "latest", "current", "new", "good", "bad",
}


def _maybe_save_city(query: str):
    """Persist city name from weather queries for future use."""
    if _LOCATION_RE.search(query):
        m = re.search(r"\bin\s+([A-Z][a-z]{2,15})\b", query)
        if m:
            candidate = m.group(1).strip()
            if candidate.lower() not in _NOT_A_CITY:
                context_store.set("city", candidate)


# ── Public entry point ────────────────────────────────────────────────────────

def fetch(query: str) -> str:
    """
    Main entry point. Always returns a speakable string.
    Never raises. Never uses LLM training data to answer.

    Follow-up questions are returned as plain strings.
    The pending query is stored internally; the next call to fetch()
    with the user's answer automatically resolves the full query.
    """
    global _pending_query

    query = query.strip()
    if not query:
        return "Sorry, I didn't catch that. Could you repeat?"

    # ── 1. Resolve pending follow-up ─────────────────────────────────────────
    if _pending_query:
        # User's current message is the answer to our follow-up question
        full_query  = f"{_pending_query} {query}"
        _pending_query = ""
        print(f"[latest_data] Follow-up resolved → {full_query!r}")
        return fetch(full_query)   # re-enter cleanly with the complete query

    # ── 2. Current datetime (injected everywhere) ─────────────────────────────
    now            = datetime.now()
    real_date_hint = now.strftime("%A, %d %B %Y, %I:%M %p")

    # ── 3. Log user turn ──────────────────────────────────────────────────────
    history.add("user", query, source=SRC_SEARCH)

    # ── 4. Recent conversation context ───────────────────────────────────────
    ctx_turns = history.context(n=10, exclude=["system"])
    ctx_text  = "\n".join(
        f"{'User' if t['role'] == 'user' else 'INNOSTAA'}: {t['content']}"
        for t in ctx_turns
    ) if ctx_turns else ""

    # ── 5. AI plans the action ────────────────────────────────────────────────
    decision_prompt = _build_decision_prompt(now)
    user_content    = (
        f"Recent conversation:\n{ctx_text}\n\nNew user message: {query}"
        if ctx_text else query
    )
    decision_raw = _groq(
        [
            {"role": "system", "content": decision_prompt},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=220,
    )
    print(f"[latest_data] Decision raw: {decision_raw}")

    # Parse JSON — strip fences, extract first {...}
    decision_raw = re.sub(r"```(?:json)?", "", decision_raw).strip()
    m = re.search(r"\{.*\}", decision_raw, re.DOTALL)
    decision_raw = m.group(0) if m else decision_raw

    try:
        decision = json.loads(decision_raw)
        if decision.get("action") not in ("search", "followup"):
            decision["action"]       = "search"
            decision["search_query"] = decision.get("search_query") or query
    except Exception as e:
        print(f"[latest_data] Decision parse error: {e} — defaulting to search")
        decision = {"action": "search", "search_query": query}

    action = decision["action"]
    print(f"[latest_data] action={action!r}  reasoning={decision.get('reasoning','')!r}")

    # ── 6. Follow-up: store pending, return question as spoken string ─────────
    if action == "followup":
        question       = decision.get("question", "Could you give me a bit more detail?")
        _pending_query = query          # remember what we were about to search
        history.add("assistant", question, source=SRC_SEARCH)
        print(f"[latest_data] Stored pending query: {query!r}")
        return question                 # caller speaks this, no sentinel needed

    # ── 7. Web search ─────────────────────────────────────────────────────────
    search_q = decision.get("search_query") or query
    print(f"[latest_data] Searching: {search_q!r}")

    is_news = bool(re.search(
        r"\b(news|headlines|updates|latest|happening|current events)\b",
        search_q, re.IGNORECASE,
    ))

    snippets = _web_search(search_q, is_news=is_news)

    # ── 8. Summarise first attempt ────────────────────────────────────────────
    if snippets:
        answer_text = _summarise(search_q, snippets, now)
        answer_text = _clean(answer_text) if answer_text else ""
    else:
        answer_text = ""

    # ── 9. Retry with rephrased query if answer was weak or empty ────────────
    if not answer_text or _is_weak_answer(answer_text):
        print(f"[latest_data] Weak answer — retrying with rephrased query")
        rephrased = _rephrase_query(query, search_q, now)
        if rephrased and rephrased.lower() != search_q.lower():
            snippets2 = _web_search(rephrased, is_news=is_news)
            if snippets2:
                answer2 = _summarise(rephrased, snippets2, now)
                answer2 = _clean(answer2) if answer2 else ""
                if answer2 and not _is_weak_answer(answer2):
                    answer_text = answer2
                    print(f"[latest_data] Retry succeeded")

    # ── 10. Final fallback if both attempts produced nothing ──────────────────
    if not answer_text or _is_weak_answer(answer_text):
        if not snippets:
            answer_text = "I was unable to reach the web right now. Please try again in a moment."
        else:
            answer_text = "I searched the web but couldn't find a clear answer for that. Try rephrasing your question."

    # ── 11. Persist city + log answer ────────────────────────────────────────
    _maybe_save_city(search_q)
    history.add("assistant", answer_text, source=SRC_SEARCH)
    return answer_text


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Context store: {context_store.summary()}")
    print(f"Current datetime: {datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        result = fetch(q)
        print(f"INNOSTAA: {result}\n")