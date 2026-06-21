import threading
import re
from datetime import datetime
from typing import Optional
 
# ── Source tag constants ──────────────────────────────────────────────────────
SRC_CASUAL   = "casual"     # main Groq casual AI   (innostaa_pyttsx3 / ai_reply)
SRC_SEARCH   = "search"     # latest_data fetch()
SRC_VISION   = "vision"     # live_video NVIDIA model
SRC_RESEARCH = "research"   # research.start()
SRC_SYSTEM   = "system"     # internal / greetings (not sent to any model)
 
# ── Filler words that are NOT topics ─────────────────────────────────────────
# (moved into _FILLER_PHRASES below, alongside noise phrases)
 
# Noise phrases stripped before topic extraction
_NOISE_PHRASES = [
    # Leading action verbs + modifiers
    r"^(now\s+)?(search(\s+about|\s+for)?|look\s+up|find(\s+info(rmation)?\s+(on|about))?|google"
    r"|tell\s+me\s+about|explain(\s+to\s+me)?|describe"
    r"|research(\s+on|\s+about|\s+more\s+about)?"
    r"|make\s+a?\s*(\w+\s+)?report\s+(on|about)"
    r"|write\s+a?\s*(\w+\s+)?report\s+(on|about)"
    r"|report\s+on|create\s+a?\s*report\s+(on|about)"
    r"|what\s+is|what\s+are|who\s+is|who\s+are|how\s+does|how\s+do"
    r"|can\s+you|please|i\s+want\s+to\s+know(\s+about)?|i\s+want\s+to\s+learn(\s+about)?"
    r"|i\s+want\s+to\s+understand|let\s+(me|us)\s+talk\s+about|let\s+us\s+discuss"
    r"|let'?s\s+(talk\s+about|discuss|look\s+at|explore|learn\s+about)"
    r"|give\s+me\s+(info(rmation)?\s+)?(on|about))\s+",
    # Trailing noise
    r"\s+on\s+the\s+internet\.?$",
    r"\s+online\.?$",
    r"\s+please\.?$",
]
_NOISE_RE = re.compile("|".join(_NOISE_PHRASES), re.IGNORECASE)
 
# Filler phrases that remain after stripping noise — these are NOT topics
_FILLER_PHRASES = {
    "it", "that", "this", "them", "they", "he", "she", "the topic",
    "the thing", "what you said", "the same", "same thing", "the subject",
    "something", "anything", "everything", "that topic", "the last topic",
    "the one", "the result", "what we discussed", "what i said",
    "about it", "on that", "on it", "about that", "the same thing",
}
 
 
class ConversationHistory:
    """
    Thread-safe, append-only log of every exchange in INNOSTAA.
 
    Each entry is a dict:
        {
            "role":      "user" | "assistant",
            "content":   str,
            "source":    one of the SRC_* constants,
            "timestamp": ISO-8601 string,
        }
    """
 
    def __init__(self, max_total: int = 120):
        """
        max_total — hard cap on how many turns are kept in memory at once.
        120 turns ≈ a full hour of heavy conversation; old turns drop off the front.
        """
        self._turns: list[dict] = []
        self._lock  = threading.Lock()
        self._max   = max_total
 
    # ── Write ─────────────────────────────────────────────────────────────────
 
    def add(self, role: str, content: str, source: str = SRC_CASUAL) -> None:
        """
        Append one conversation turn.
 
        Parameters
        ----------
        role    : "user" or "assistant"
        content : the spoken / typed text
        source  : SRC_CASUAL | SRC_SEARCH | SRC_VISION | SRC_RESEARCH | SRC_SYSTEM
        """
        if not content or not content.strip():
            return
        entry = {
            "role":      role,
            "content":   content.strip(),
            "source":    source,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock:
            self._turns.append(entry)
            if len(self._turns) > self._max:
                # Drop oldest turns in pairs (user+assistant) to keep history coherent
                self._turns = self._turns[-self._max:]
 
    # ── Read ──────────────────────────────────────────────────────────────────
 
    def context(
        self,
        n:       int            = 20,
        source:  Optional[str]  = None,
        exclude: Optional[list] = None,
    ) -> list[dict]:
        """
        Return the last *n* turns as a list of {"role": ..., "content": ...} dicts
        suitable for passing directly into any Groq / OpenAI messages list.
 
        Parameters
        ----------
        n       : max turns to return (most recent)
        source  : if given, filter to only turns from that source
        exclude : list of source tags to exclude (e.g. [SRC_SYSTEM])
 
        Returns dicts WITHOUT the "source" and "timestamp" keys — just role+content,
        so they can be dropped straight into an API messages list.
        """
        exclude = exclude or [SRC_SYSTEM]
        with self._lock:
            turns = list(self._turns)
 
        if source:
            turns = [t for t in turns if t["source"] == source]
        turns = [t for t in turns if t["source"] not in exclude]
        turns = turns[-n:]
        return [{"role": t["role"], "content": t["content"]} for t in turns]
 
    def raw(self, n: int = 40) -> list[dict]:
        """Return last *n* full entries (with source + timestamp). Useful for debug."""
        with self._lock:
            return list(self._turns[-n:])
 
    # ── Topic resolution ──────────────────────────────────────────────────────
 
    def get_last_topic(self, window: int = 14) -> str:
        """
        Scan the last *window* USER turns backwards and return the most recent
        concrete topic the user mentioned.
 
        Strips filler words and noise phrases so "search about it" resolves to
        whatever the user was asking about before, not "it".
 
        Returns "" if nothing concrete is found.
 
        Examples
        --------
        User: "tell me about quantum computing"
        User: "now search about it on the internet"
        → get_last_topic() → "quantum computing"
 
        User: "let's discuss black holes"
        User: "make a report on that"
        → get_last_topic() → "black holes"
        """
        with self._lock:
            # Collect last window USER turns (most recent first)
            user_turns = [
                t["content"]
                for t in reversed(self._turns)
                if t["role"] == "user" and t["source"] != SRC_SYSTEM
            ][:window]
 
        for text in user_turns:
            candidate = self._extract_topic(text)
            if candidate:
                return candidate
        return ""
 
    def get_last_user_text(self) -> str:
        """Return the most recent user utterance (any source)."""
        with self._lock:
            for t in reversed(self._turns):
                if t["role"] == "user":
                    return t["content"]
        return ""
 
    def get_context_summary(self, n: int = 6) -> str:
        """
        Return a short plain-text summary of the last *n* turns,
        useful for injecting into a system prompt.
 
        Format:
            [casual] User: ...
            [casual] INNOSTAA: ...
            [vision] User: ...
        """
        with self._lock:
            recent = list(self._turns[-n:])
        lines = []
        for t in recent:
            tag  = f"[{t['source']}]"
            who  = "User" if t["role"] == "user" else "INNOSTAA"
            lines.append(f"{tag} {who}: {t['content']}")
        return "\n".join(lines)
 
    # ── Internal helpers ──────────────────────────────────────────────────────
 
    @staticmethod
    def _extract_topic(text: str) -> str:
        """
        Strip noise phrases and filler from a user utterance and return
        what's left as the candidate topic.  Returns "" if the remainder
        is empty or a known filler word / phrase.
        """
        text = text.strip()
 
        # Iteratively strip leading noise phrases (may stack, e.g. "now search about")
        prev = None
        while prev != text:
            prev = text
            text = _NOISE_RE.sub("", text).strip()
 
        # Remove trailing punctuation
        text = text.rstrip("?.!,;:")
 
        # Reject if empty, too short, or a known filler
        if not text or len(text) < 3:
            return ""
        if text.lower() in _FILLER_PHRASES:
            return ""
 
        # Reject if the first word is a vague pronoun
        first_word = text.lower().split()[0]
        if first_word in {"it", "that", "this", "them", "they", "the"}:
            # Allow "the <noun>" patterns — only reject single-word pronouns
            if len(text.split()) == 1 or first_word in {"it", "that", "this", "them", "they"}:
                return ""
        print (type(text))
        return text
 
 
# ── Module-level singleton ────────────────────────────────────────────────────
# Import this object in every module:
#     from conversation_history import history
history = ConversationHistory(max_total=120) 