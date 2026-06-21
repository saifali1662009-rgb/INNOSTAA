import os
import time

# ── API key bootstrap — runs before any heavy imports ─────────────────────────
def _bootstrap_env():
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(_env_file):
        return
    with open(_env_file, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip(); _v = _v.strip().strip('"').strip("'")
            if _k and _v and _k not in os.environ:
                os.environ[_k] = _v
_bootstrap_env()
# ─────────────────────────────────────────────────────────────────────────────

import wave
import unicodedata
import queue
import requests
import pytesseract
import barge_in
#import pyautogui
import threading
import mouse
import tictactoe as ttt_module
import research
import automation
import whiteboard as wb_module
import cv2
import base64
from datetime import datetime
#from fuzzywuzzy import fuzz
import keyboard
#import edge_tts
#import asyncio
import latest_data
#import pygame
#import io
#from claude_gui import InnostaaWithGUI
#from pywinauto import Desktop
#from pywinauto.findwindows import ElementNotFoundError
import numpy as np
import sounddevice as sd
#import soundfile as sf
import subprocess
#import shutil
import json
import random
import psutil
from groq import Groq
import re
import sys
import live_video
import file_opener
import tempfile
import webview
# ── NEW: unified conversation history ─────────────────────────────────────────
from conversation_history import history, SRC_CASUAL, SRC_SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
# ── Global state ──────────────────────────────────────────────────────────────
window               = None
SHUTDOWN_REQUESTED   = False
MEMORY_DIR           = os.path.join(BASE_DIR, "memory")
#MEMORY_DIR           = os.path.join(os.path.dirname(__file__), "memory")
MEMORY_VERSIONS_DIR  = os.path.join(MEMORY_DIR, "versions")
ACTIVE_MEMORY_FILE   = os.path.join(MEMORY_DIR, "user_memory.txt")
os.makedirs(MEMORY_VERSIONS_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

audio_stream         = None
is_speaking          = threading.Event()
last_typed_char      = ""
mouse_thread         = None
gesture_mouse_active = False
vision_thread        = None
VIDEO_MODE           = None
ttt_game             = ttt_module.HandGestureTicTacToe()
wb_game              = wb_module.GestureWhiteboard()
cv2_frame_running    = False
text_input_queue     = queue.Queue()
is_processing        = threading.Lock()
TEXT_MODE_ACTIVE     = False
last_query           = ""
typing_mode          = False
gui_manager          = None
ACTIVE_GESTURE       = None
gesture_video_running = False
gesture_video_thread  = None

EXIT_KEYWORDS    = ["bye", "exit", "quit", "see you later", "good bye", "end conversation"]
UTILITY_KEYWORDS = ["weather", "time", "date", "day", "capital", "define", "meaning", "solve", "calculate"]

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"

SAMPLE_RATE  = 16000
BLOCKSIZE    = 4_000

SPEECH_THRESHOLD = 230
SILENCE_LIMIT    = 4
MIN_SPOKEN_BLOCKS = 3

# Short-term session history (kept for backward-compat with reflect_and_write_memory)
CONVERSATION_HISTORY = []
MAX_MEMORY = 6

# ── Date / time helpers ───────────────────────────────────────────────────────

def get_current_datetime():
    return datetime.now()

def get_current_date_str():
    return get_current_datetime().strftime("%d %B %Y")

def get_current_day():
    return get_current_datetime().strftime("%A")

def get_current_time_str():
    return get_current_datetime().strftime("%I:%M %p")

def get_current_year():
    return get_current_datetime().year

# ── Groq client ───────────────────────────────────────────────────────────────

def _make_groq_client():
    key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please run INNOSTAA via run.py to set up your API keys."
        )
    return Groq(api_key=key)

groq_client = None

def get_groq_client():
    global groq_client
    if groq_client is None:
        groq_client = _make_groq_client()
    return groq_client

# ── Audio queue ───────────────────────────────────────────────────────────────
audio_q = queue.Queue()

# ── Motivational greetings ────────────────────────────────────────────────────
MOTIVATIONAL_GREETINGS = [
    "Once a legend said mistakes build experience, experience builds success, and success builds the future. So tell me, what are we building today?",
    "Every champion was once a beginner who refused to quit. So, what are we not quitting today?",
    "A new day means a new chance to grow stronger. How can I support your journey today?",
    "believe you can and you are halfway there. so today, where are we moving towards?",
    "people works better when they know what the goal is and why. so, what is you goal?",
    "the best brains of the nation may found on the last benches of the classroom. are you that back bencher?",
    "the best way to not feel hopeless is to get up and do something. so, what are we doing today?",
    "the best source of knowledge is experience. so, what are we experiencing today?",
    "A great person said I never loose. I either win or learn. As he removed loose word from his dictionary.",
    "Today is tough, tomorrow is much tougher, but the day after tomorrow is beautiful and many people give up tomorrow evening. So, do not give up, and always move ahead."
]

# ── Utilities ─────────────────────────────────────────────────────────────────

def safe_print(*args):
    """Print safely: handles None stdout (--noconsole builds) and encoding errors."""
    if sys.stdout is None:
        return
    try:
        print(*args)
    except Exception:
        fixed = []
        for a in args:
            try:
                fixed.append(str(a))
            except Exception:
                fixed.append(str(a).encode("ascii", "ignore").decode())
        try:
            print(" ".join(fixed))
        except Exception:
            pass


def audio_callback(indata, frames, time_info, status):
    if not MIC_ENABLED:
        return
    raw = bytes(indata)
    audio_q.put(raw)                    # always feed — listen() handles is_speaking internally
    if is_speaking.is_set():            # tee to barge_in during TTS
        try:
            import barge_in as _bi
            _bi._barge_q.put_nowait(raw)
        except Exception:
            pass
    else:                               # tee to calib queue when mic is idle/listening
        try:
            import barge_in as _bi
            _bi._calib_q.put_nowait(raw)
        except Exception:
            pass


_tts_lock  = threading.Lock()
EDGE_VOICE = "en-US-AriaNeural"
# Other voices: en-US-AnaNeural, en-US-GuyNeural, en-US-JennyNeural,
#               en-US-MichelleNeural, en-US-RogerNeural, en-US-SteffanNeural


def speak(text):
    barge_in.speak(text)
#    """Speak text aloud via Edge TTS and push it to the GUI chat bubble.
#    Also logs the assistant turn to the unified conversation history."""
#    global window
#    print("Assistant:", text)
#
#    # ── Log assistant turn to unified history ─────────────────────────────────
#    history.add("assistant", text, source=SRC_SYSTEM)
#    # ─────────────────────────────────────────────────────────────────────────
#
#    safe_text = (
#        text.replace("'", "\\'")
#            .replace('"', '\\"')
#            .replace("\n", " ")
#            .replace("\r", "")
#    )
#    if window:
#        try:
#            window.evaluate_js(f"updateStatus('speaking', '{safe_text}')")
#            window.evaluate_js(f"addMessage('assistant', '{safe_text}')")
#        except Exception:
#            pass
#
#    is_interrupted.clear()
#    is_speaking.set()
#
#    def _interrupt_watcher():
#        INTERRUPT_THRESHOLD = 60
#        INTERRUPT_BLOCKS    = 4
#        count = 0
#        ensure_audio_stream()
#        while is_speaking.is_set() and MIC_ENABLED:
#            try:
#                block   = audio_q.get(timeout=0.05)
#                samples = np.frombuffer(block, dtype=np.int16)
#                energy  = int(np.abs(samples).mean())
#                if energy > INTERRUPT_THRESHOLD:
#                    count += 1
#                    if count >= INTERRUPT_BLOCKS:
#                        is_interrupted.set()
#                        is_speaking.clear()
#                        sd.stop()
#                        return
#                else:
#                    count = 0
#            except Exception:
#                pass
#
#    watcher = threading.Thread(target=_interrupt_watcher, daemon=True)
#    watcher.start()
#
#    with _tts_lock:
#        try:
#            async def _synthesize():
#                communicate = edge_tts.Communicate(text, voice=EDGE_VOICE)
#                audio_data  = b""
#                async for chunk in communicate.stream():
#                    if chunk["type"] == "audio":
#                        if is_interrupted.is_set():
#                            break
#                        audio_data += chunk["data"]
#                return audio_data
#
#            audio_data = asyncio.run(_synthesize())
#
#            if not is_interrupted.is_set() and audio_data:
#                audio_io = io.BytesIO(audio_data)
#                data, samplerate = sf.read(audio_io)
#                sd.play(data, samplerate)
#                sd.wait()
#
#        except Exception as e:
#            print("TTS Error:", e)
#
#    is_speaking.clear()
#    watcher.join(timeout=0.3)
#
#    while not audio_q.empty():
#        try:
#            audio_q.get_nowait()
#        except Exception:
#            break
#
#    if window:
#        try:
#            if MIC_ENABLED:
#                window.evaluate_js("updateStatus('listening', '')")
#            else:
#                window.evaluate_js("updateStatus('idle', '')")
#        except Exception:
#            pass


def smart_listen():
    """Follow-up input: uses text queue in text mode, voice otherwise."""
    if TEXT_MODE_ACTIVE:
        if window:
            try:
                window.evaluate_js("updateStatus('listening', '')")
            except Exception:
                pass
        try:
            result = text_input_queue.get(timeout=30)
            if window:
                try:
                    safe = (result.replace("'", "\\'")
                                  .replace('"', '\\"')
                                  .replace("\n", " "))
                    window.evaluate_js(f"addMessage('user', '{safe}')")
                except Exception:
                    pass
            return result
        except Exception:
            return ""
    else:
        return listen()

# ── Text normalisation ────────────────────────────────────────────────────────

def normalize_text(text):
    try:
        cleaned = unicodedata.normalize("NFKD", text)
        cleaned = cleaned.encode("ascii", "ignore").decode()
        replacements = {
            "−": "-", "–": "-", "—": "-",
            "·": "*", "×": "*", "÷": "/",
            "√": "sqrt ", "π": "pi ", "∞": "infinity ",
            "≤": "<=", "≥": ">=", "≠": "!=",
        }
        for k, v in replacements.items():
            cleaned = cleaned.replace(k, v)
        cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()
    except Exception:
        return text


def normalize_search_query(query):
    q = query.lower().strip()
    if q in ["latest news", "search news", "news"]:
        return "latest global news headlines today"
    if "tech" in q or "technology" in q or "electronics" in q:
        return "latest technology and electronics news today"
    if "ai" in q or "artificial intelligence" in q:
        return "latest artificial intelligence news today"
    return query


def parse_date_from_text(text):
    try:
        return datetime.strptime(text, "%d %B").replace(year=get_current_year())
    except Exception:
        return None


def clean_for_voice(text):
    """Strip markdown and fix symbols so TTS reads cleanly."""
    text = text.replace("**", "")
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", text)
    text = re.sub(r"(\d+)\s*°?\s*C\b", r"\1 degrees Celsius", text)
    text = re.sub(r"(\d+)\s*°?\s*F\b", r"\1 degrees Fahrenheit", text)
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)
    text = " ".join(text.split())
    return text.strip()

# ── Memory ────────────────────────────────────────────────────────────────────

def update_memory(role, content):
    """Log a turn to both the short-term session list AND the unified history."""
    CONVERSATION_HISTORY.append({"role": role, "content": content})
    if len(CONVERSATION_HISTORY) > MAX_MEMORY:
        CONVERSATION_HISTORY[:] = CONVERSATION_HISTORY[-MAX_MEMORY:]

    # ── Unified history: only log user turns here; assistant turns are logged
    #    inside speak() so every spoken word — including greetings, system
    #    messages, search responses — is captured too.
    if role == "user":
        history.add("user", content, source=SRC_CASUAL)


# ── AI-powered fallback command detector ─────────────────────────────────────
# Maps every intent label that process() knows about → exact string process()
# checks with `in intent`.  The AI must return one of these keys or "none".
_INTENT_LABELS = {
    "open":               "open {arg}",
    "close":              "close {arg}",
    "search for latest data": "search for latest data",
    "exit":               "exit",
    "turning on light":   "turning on light",
    "turning off light":  "turning off light",
    "press keyboard":     "press keyboard {arg}",
    "use virtual mouse":  "use virtual mouse",
    "stop virtual mouse": "stop virtual mouse",
    "use whiteboard":     "use whiteboard",
    "stop whiteboard":    "stop whiteboard",
    "type":               "type",
    "play tic tac toe":   "play tic tac toe",
    "stop tic tac toe":   "stop tic tac toe",
    "reset game":         "reset game",
    "start live video":   "start live video",
    "stop live video":    "stop live video",
    "start share screen": "start share screen",
    "stop share screen":  "stop share screen",
    "intelligent file opener": "intelligent file opener",
    "research":           "research {arg}",
    "automation mode":    "automation mode",
    "none":               "none",
}
def _ai_command_check(user_text: str) -> str:
    def _history_context(n=10):
        try:
            ctx = history.context(n=n)
            return "\n".join(
                f"{m['role']}: {m['content']}"
                for m in ctx
            )
        except Exception:
            return ""
    history_ctx = _history_context(10)    
    _FALLBACK_SYSTEM = """ You are the command-detection brain of INNOSTAA, a voice assistant.
    above are the current conversation history between the user and the assistant. Based on this conversation history, and the user's latest message, decide if the user is trying to trigger any of the following specific INNOSTAA actions (not casual talk). If so, reply with ONLY a JSON object containing the EXACT key from the list below that best matches the user's intent.
    The main intent classifier already decided this input is "casual conversation",
    but it sometimes makes mistakes.  Your job is a SECOND OPINION CHECK, it don't have history of the session so, you have to think what user wants:
    decide whether the user is actually asking for a specific INNOSTAA action.
    
    Available actions (use EXACTLY these keys):
      open {app}            — open an application
      close {app}           — close an application
      search for latest data — get live/current info (news, prices, weather, current leaders)
      exit                  — user wants to say goodbye and quit
      turning on light      — turn room light on
      turning off light     — turn room light off
      press keyboard {key}  — press a keyboard key
      use virtual mouse     — start gesture mouse
      stop virtual mouse    — stop gesture mouse
      use whiteboard        — start gesture whiteboard
      stop whiteboard       — stop gesture whiteboard
      play tic tac toe      — start tic tac toe game
      stop tic tac toe      — stop tic tac toe game
      reset game            — reset tic tac toe game
      start live video      — start camera vision mode
      stop live video       — stop camera vision mode
      start share screen    — start screen-share vision mode
      stop share screen     — stop screen-share vision mode
      intelligent file opener — open a file, folder, or drive
      research {topic}      — deep research report on a topic
      automation mode       — automate tasks (email, file management, etc.)
      none                  — it really is just casual conversation
    
    Rules:
    - Reply with ONLY a JSON object: {"cmd": "<key>"}
    - you need to differentiate wheather the user using the words like "search", "open", "close", "report","emails", "research", etc to make an action or just using these words to complete his casual conversation. so, seriously choose the action only when the user really wants to do it, otherwise reply with {"cmd": "none"}.
    - Replace {app}, {key}, {topic} with the actual value from the user's message.
    - If the user is asking HOW to do something (not actually doing it), reply {"cmd": "none"}.
    - If genuinely unsure → {"cmd": "none"}.
    - think about the users tone seriously wheather he wants to talk or he wants to do any action
    - No explanation. No markdown. Just the raw JSON object.
    
    Examples:
      "launch spotify for me"        → {"cmd": "open spotify"}
      "can you kill chrome"          → {"cmd": "close chrome"}
      "what's the news today"        → {"cmd": "search for latest data"}
      "i need to go now bye"         → {"cmd": "exit"}
      "switch the light on please"   → {"cmd": "turning on light"}
      "start the camera"             → {"cmd": "start live video"}
      "do research on black holes"   → {"cmd": "research black holes"}
      "how do I open spotify"        → {"cmd": "none"}
      "tell me a joke"               → {"cmd": "none"}
    """    
    
    import json as _json

    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY') or GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": "openai/gpt-oss-120b",   
        "messages": [
            {"role": "system", "content": f"{history_ctx}\n{_FALLBACK_SYSTEM}"},
            {"role": "user",   "content": user_text},
        ],
        "temperature": 0.0,               # deterministic
        "max_tokens":  60,                # we only need a short JSON blob
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"[_ai_command_check] HTTP {resp.status_code}")
            return "none"

        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()

        data = _json.loads(raw)
        cmd  = data.get("cmd", "none").strip()

        if cmd and cmd != "none":
            print(f"[_ai_command_check] Fallback detected command: {cmd!r}")
        return cmd

    except Exception as e:
        print(f"[_ai_command_check] Error: {e}")
        return "none"

# ─────────────────────────────────────────────────────────────────────────────


def ai_reply(user_text):
    """Generate a casual AI reply and log both turns."""
    try:
        update_memory("user", user_text)

        user_memory_text = ""
        if os.path.exists(ACTIVE_MEMORY_FILE):
            with open(ACTIVE_MEMORY_FILE, "r", encoding="utf-8") as f:
                user_memory_text = f.read().strip()

        lower_text   = user_text.lower()
        allow_memory = not any(word in lower_text for word in UTILITY_KEYWORDS)

        current_day  = get_current_day()
        current_date = get_current_date_str()
        current_time = get_current_time_str()
        current_year = get_current_year()

        system_prompt = f"""SYSTEM CONTEXT:
- Today is {current_day}.
- Today's date is {current_date}.
- Current time is {current_time}.
- Current year is {current_year}.

Your name is INNOSTAA, made by Saif for voice assistance.
You can turn on or off a room light (but not adjust brightness), do casual talk,
search on the internet, write or open applications when asked, can do research, right emails. Your best feature is
a memory system that keeps user details privately on their PC.

Respond clearly and friendly. You are on a VOICE conversation — no special
characters like |, :, -, bullet points, or markdown in your answers.
Your knowledge is limited to 2023. For live/current information the system
handles it separately — never say you are "checking the internet". if get context about any latest data and user want any comment about it just answer that and never say that i don't have current data.
You only speak English. If you receive non-English input, ignore it briefly.
If a last study topic is in memory, suggest continuing it during casual chat.
You are weak at calendar calculations; mention this and ask the user to recheck.
never be confident on data you have, you have data till 2023 while the current year is different, you don't know the current stats, info about the global leader and anything that is rlated to time line, also, there will be many events that held after the year of data you have. so, you can simply say you don't have knowledge but you can search on internet.
Known background information about the user:
{user_memory_text if (allow_memory and user_memory_text) else "No background information should be used for this query."}

INSTRUCTIONS:
- Use background information naturally only when relevant; never mention memory.
- Do NOT say phrases like "I remember" or "from memory".
- Keep responses concise: 1-3 short sentences preferred.
- Ask at most one follow-up question if more detail would help.
- Remind about events from memory when appropriate; never invent reminders.
- Never describe your own origin or purpose unless explicitly asked."""

        # ── Build messages from unified history so ALL modules share context ──
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history.context(n=14))
        # ─────────────────────────────────────────────────────────────────────

        completion = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages
        )
        reply = completion.choices[0].message.content.strip()
        reply = normalize_text(reply)

        # Log assistant turn to short-term session list only;
        # speak() will log it to unified history automatically.
        CONVERSATION_HISTORY.append({"role": "assistant", "content": reply})
        if len(CONVERSATION_HISTORY) > MAX_MEMORY:
            CONVERSATION_HISTORY[:] = CONVERSATION_HISTORY[-MAX_MEMORY:]

        return reply

    except Exception as e:
        print(f"[ai_reply] Error: {e}")
        return "Sorry, I faced a small issue while responding."


def reflect_and_write_memory():
    """Run once at session end — reflects on conversation and rewrites memory file."""
    old_memory = ""
    if os.path.exists(ACTIVE_MEMORY_FILE):
        with open(ACTIVE_MEMORY_FILE, "r", encoding="utf-8") as f:
            old_memory = f.read()

    reflection_prompt = f"""You are a reflective personal assistant.

Existing user memory:
{old_memory if old_memory else "No prior memory."}

Conversation transcript:
{CONVERSATION_HISTORY}

TASK:
- Extract ONLY essential, stable facts explicitly stated by the user.
- Ignore casual talk, jokes, greetings.
- Never write anything like "**new memories**", "**updated memories**" or similar.
- Write only things related to the user giving their details.
- Categorise information: interests, hobbies, personal details, important dates,
  preferences, routines, goals, relationships, work/study, health, reminders etc.
- For studies: only write the last topic studied and in which subject. Homework,
  studies, and exam prep go under work/study.
- Never write anything related to shopping or money.
- Do NOT guess or assume anything.
- Do not remove any memory unless given an explicit command to do so.
- Rewrite memory combining old and new important facts.
- Only remove an old memory item if it was updated during this session.
- Rewrite as clean, short paragraphs.
- Strictly never write something not stated by the user.
- Do NOT mention the assistant or this reflection process.

Output ONLY the rewritten memory text."""

    try:
        completion = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": reflection_prompt}]
        )
        new_memory = completion.choices[0].message.content.strip()

        timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        version_file = os.path.join(MEMORY_VERSIONS_DIR, f"memory_{timestamp}.txt")

        if old_memory.strip():
            with open(version_file, "w", encoding="utf-8") as f:
                f.write(old_memory)

        with open(ACTIVE_MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write(new_memory)

    except Exception as e:
        print(f"[reflect_and_write_memory] Error: {e}")


def shutdown_assistant():
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    try:
        if gui_manager:
            gui_manager.root.after(200, gui_manager.root.destroy)
    except Exception:
        pass
    time.sleep(0.5)
    os._exit(0)

# ── Audio stream ──────────────────────────────────────────────────────────────

_persistent_stream = None

def ensure_audio_stream():
    """Open the mic stream once and keep it alive for the whole session."""
    global _persistent_stream, audio_stream
    if _persistent_stream is None or not _persistent_stream.active:
        try:
            if _persistent_stream is not None:
                try:
                    _persistent_stream.stop()
                    _persistent_stream.close()
                except Exception:
                    pass
            _persistent_stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                dtype="int16",
                channels=1,
                callback=audio_callback
            )
            _persistent_stream.start()
            audio_stream = _persistent_stream
        except Exception as e:
            print(f"[ensure_audio_stream] Error: {e}")
    return _persistent_stream


def listen():
    global audio_stream, text
    safe_print("Listening...")

    # Use the calibrated noise floor from barge_in so the threshold
    # adapts to the actual room/mic — no more "mouth near keyboard" problem.
    import barge_in as _bi
    _calib_threshold = _bi._dyn_threshold   # already noise_floor × NOISE_MULT
    START_THRESHOLD   = max(int(_calib_threshold * 0.7), 100)  # slightly below barge-in
    STOP_THRESHOLD    = max(int(_calib_threshold * 0.35), 50)
    MIN_SPEECH_BLOCKS = 3
    SILENCE_TIME      = 0.75

    audio_frames     = []
    speaking_started = False
    last_voice_time  = time.time()

    # After barge-in: instead of discarding the audio captured during TTS,
    # recover the voiced frames that barge_in.py already confirmed as speech.
    # This preserves the user's first words so Whisper gets the full utterance.
    if is_interrupted.is_set():
        is_interrupted.clear()
        # Pull in the voiced frames that triggered the barge-in
        import barge_in as _bi
        if _bi.barge_audio_frames:
            audio_frames     = list(_bi.barge_audio_frames)
            speaking_started = True                  # speech already in progress
            last_voice_time  = time.time()
            _bi.barge_audio_frames = []
            print(f"[listen] Recovered {len(audio_frames)} barge-in frames", flush=True)
        # Discard echo-contaminated blocks that arrived after the interrupt
        while True:
            try: audio_q.get_nowait()
            except Exception: break
        # Brief pause so any remaining TTS echo fades before we resume capture
        time.sleep(0.10)
    else:
        # Normal start after TTS — flush stale audio.
        # audio_callback already stopped teeing to barge_q, but the speaker
        # echo may still be arriving in audio_q for a brief moment.
        # Drain anything that arrived in the last ~150 ms to avoid self-listen.
        time.sleep(0.15)
        while True:
            try: audio_q.get_nowait()
            except Exception: break

    ensure_audio_stream()

    while True:
        if not MIC_ENABLED:
            return ""

        if is_speaking.is_set():
            time.sleep(0.01)
            continue

        try:
            block = audio_q.get(timeout=0.2)
        except Exception:
            continue

        samples = np.frombuffer(block, dtype=np.int16)
        energy  = int(np.abs(samples).mean())

        if not speaking_started:
            if energy > START_THRESHOLD:
                speaking_started = True
                audio_frames.append(block)
                last_voice_time  = time.time()
            continue

        audio_frames.append(block)

        if energy > STOP_THRESHOLD:
            last_voice_time = time.time()

        if time.time() - last_voice_time > SILENCE_TIME:
            break

    if len(audio_frames) < MIN_SPEECH_BLOCKS:
        safe_print("(Ignored noise)")
        return ""

    temp_path = os.path.join(tempfile.gettempdir(), f"ania_{time.time()}.wav")
    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(audio_frames))

    try:
        with open(temp_path, "rb") as f:
            resp = get_groq_client().audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo",
                language="en",
                prompt="INNOSTAA assistant. User speaks clear English.",
            )
        os.remove(temp_path)
        text = normalize_text(resp.text.strip())
        safe_print("You:", text)
        return text
    except Exception as e:
        safe_print("STT Error:", e)
        return ""

# ── App control ───────────────────────────────────────────────────────────────

def launch_app(app_name):
    app_name = app_name.lower().strip()

    fixed_paths = {
        "ed":                  r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "edge":                r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "microsoft edge":      r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "chrome":              r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "google chrome":       r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "file explorer":       r"C:\\Windows\\explorer.exe",
        "explorer":            r"C:\\Windows\\explorer.exe",
        "word":                r"C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        "microsoft word":      r"C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        "whatsapp":            r"C:\\Users\\{}\\AppData\\Local\\WhatsApp\\WhatsApp.exe".format(os.getlogin()),
        "spotify":             r"C:\\Users\\{}\\AppData\\Roaming\\Spotify\\Spotify.exe".format(os.getlogin()),
        "perplexity":          r"C:\\Users\\{}\\AppData\\Local\\Programs\\Perplexity\\Perplexity.exe".format(os.getlogin()),
        "vs code":             r"C:\\Users\\{}\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe".format(os.getlogin()),
        "visual studio code":  r"C:\\Users\\{}\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe".format(os.getlogin()),
    }

    for key in fixed_paths:
        if key in app_name:
            path = fixed_paths[key]
            if os.path.isfile(path):
                subprocess.Popen(path)
                return f"Opening {key}."
            else:
                return f"I found {key}, but the installation seems missing."

    built_in = {
        "notepad":    "notepad",
        "calculator": "calc",
        "cmd":        "cmd",
        "powershell": "powershell",
        "paint":      "mspaint",
        "settings":   "ms-settings:",
    }
    for key, cmd in built_in.items():
        if key in app_name:
            subprocess.Popen(cmd, shell=True)
            return f"Opening {key}."

    search_paths = [
        r"C:\\Program Files",
        r"C:\\Program Files (x86)",
        r"C:\\Users\\{}\\AppData\\Local".format(os.getlogin()),
    ]
    for path in search_paths:
        for root, dirs, files in os.walk(path):
            for file in files:
                if (file.lower().endswith(".exe") and
                        app_name.replace(" ", "") in file.lower().replace(" ", "")):
                    subprocess.Popen(os.path.join(root, file))
                    return f"Opening {app_name}."

    return f"I couldn't find an application named {app_name}."


def get_active_app():
    try:
        import win32gui
        import win32process
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return psutil.Process(pid).name().lower()
    except Exception:
        return ""


def close_app(app_name):
    app_name = app_name.lower().strip().replace(" ", "")

    process_map = {
        "chrome":           "chrome.exe",
        "googlechrome":     "chrome.exe",
        "edge":             "msedge.exe",
        "microsoftedge":    "msedge.exe",
        "whatsapp":         "WhatsApp.exe",
        "spotify":          "Spotify.exe",
        "fileexplorer":     "explorer.exe",
        "explorer":         "explorer.exe",
        "cmd":              "cmd.exe",
        "commandprompt":    "cmd.exe",
        "vs":               "Code.exe",
        "vscode":           "Code.exe",
        "visualstudiocode": "Code.exe",
        "perplexity":       "Perplexity.exe",
    }

    target_proc = None
    for key, value in process_map.items():
        if key in app_name:
            target_proc = value

    if not target_proc:
        target_proc = app_name + ".exe"

    closed_any = False
    for proc in psutil.process_iter(["name"]):
        try:
            if target_proc.lower() in proc.info["name"].lower():
                proc.terminate()
                closed_any = True
        except Exception:
            pass

    return f"Closing {app_name}." if closed_any else f"I couldn't find {app_name} running."

# ── Keyboard helpers ──────────────────────────────────────────────────────────

def press_key(key_name):
    key_map = {
        "enter": "enter", "enter.": "enter",
        "space": "space", "escape": "esc",
        "backspace": "backspace", "delete": "delete",
        "coma": ",", "qoute": "'", "double qoute": '"',
        "full stop": ".", "question mark": "?",
        "exclamation mark": "!", "colon": ":", "semicolon": ";",
        "slash": "/", "backslash": "\\", "dash": "_", "hyphen": "-",
        "ctrl c": ["ctrl", "c"], "control c": ["ctrl", "c"],
        "ctrl v": ["ctrl", "v"], "control v": ["ctrl", "v"],
        "ctrl x": ["ctrl", "x"], "control x": ["ctrl", "x"],
    }
    key_name = key_name.lower().strip()
    if key_name in key_map:
        val = key_map[key_name]
        if isinstance(val, list):
            keyboard.press(val[0])
            keyboard.press(val[1])
            keyboard.release(val[1])
            keyboard.release(val[0])
        else:
            keyboard.press_and_release(val)
        return f"Pressed {key_name}."
    return f"I couldn't recognize the key {key_name}."


def convert_speech_to_keys(text):
    global last_typed_char

    command_keys = {
        "backspace": "backspace", "delete": "delete",
        "enter": "enter", "new line": "enter", "space": "space",
    }
    punctuation = {
        "full stop": ".", "period": ".", "comma": ",",
        "question mark": "?", "exclamation mark": "!",
        "colon": ":", "semicolon": ";",
    }

    text = text.lower().strip()

    if text in command_keys:
        keyboard.press_and_release(command_keys[text])
        return ""

    for word, symbol in punctuation.items():
        if word in text:
            keyboard.write(symbol + " ")
            last_typed_char = symbol
            return ""

    keyboard.write(text + " ")
    last_typed_char = text[-1]
    return ""

# ── IoT / light control ───────────────────────────────────────────────────────

ESP32_IP        = "http://10.100.20.29"
IOT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "automation_data", "iot_config.json"
)

def _load_esp32_ip():
    global ESP32_IP
    try:
        if os.path.exists(IOT_CONFIG_FILE):
            import json as _json
            with open(IOT_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if data.get("esp32_ip"):
                ESP32_IP = data["esp32_ip"]
    except Exception as e:
        print(f"IoT config load error: {e}")

_load_esp32_ip()

def check_esp32():
    try:
        r = requests.get(f"{ESP32_IP}/ping", timeout=3.0)  # was 1.2
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"[ESP32] Connection refused at {ESP32_IP}")
        return False
    except requests.exceptions.Timeout:
        print(f"[ESP32] Timeout reaching {ESP32_IP}")
        return False
    except Exception as e:
        print(f"[ESP32] Unexpected error: {e}")
        return False


def control_light(state):
    print(f"[control_light] Using ESP32_IP = {ESP32_IP}")
    if not check_esp32():
        return "ESP32 is not connected. Please check the device."
    try:
        requests.get(f"{ESP32_IP}/light/{state}", timeout=1.2)
        return f"Light turned {state}."
    except Exception:
        return "I lost connection to the ESP32."

# ── Media control ─────────────────────────────────────────────────────────────

def media_control(action):
    action  = action.lower().strip()
    actions = {
        "play": "play/pause media", "pause": "play/pause media",
        "stop": "stop media", "next": "next track",
        "next song": "next track", "previous": "previous track",
        "previous song": "previous track", "volume up": "volume up",
        "volume down": "volume down", "mute": "volume mute",
    }
    if action in actions:
        try:
            keyboard.press_and_release(actions[action])
            return f"Media {action} executed."
        except Exception as e:
            safe_print(f"Media control error: {e}")
            return f"Could not execute {action}."
    return f"Unknown media command: {action}"

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

def _read_env() -> dict:
    result = {}
    if not os.path.exists(ENV_FILE):
        return result
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip().strip('"').strip("'")
    return result

def _write_env(data: dict):
    existing = _read_env()
    existing.update(data)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        for k, v in existing.items():
            f.write(f'{k}="{v}"\n')
# ── Gesture / vision helpers ──────────────────────────────────────────────────

def cv2_frame_sender():
    global cv2_frame_running
    while cv2_frame_running:
        frame = None
        if ACTIVE_GESTURE == "tictactoe" and ttt_game.running:
            frame = ttt_game.get_frame()
        elif ACTIVE_GESTURE == "whiteboard" and wb_game.running:
            frame = wb_game.get_frame()
        elif ACTIVE_GESTURE == "mouse":
            frame = mouse.get_latest_frame()

        if frame is not None and window:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            b64    = base64.b64encode(buf).decode("utf-8")
            try:
                window.evaluate_js(f"updateVideoFrame('{b64}')")
            except Exception:
                pass
        time.sleep(0.05)

    if window:
        try:
            window.evaluate_js("clearVideoFeed()")
        except Exception:
            pass


def start_vision(mode="camera"):
    global ACTIVE_GESTURE, gesture_video_running, gesture_video_thread, gui_manager

    if ACTIVE_GESTURE == "vision":
        live_video.start_vision(mode)
        return f"Switched to {mode} mode."

    stop_active_gesture()
    live_video.start_vision(mode)
    ACTIVE_GESTURE = "vision"

    if gui_manager:
        gui_manager.set_gesture_state("vision")

    if not gesture_video_running:
        gesture_video_running = True
        gesture_video_thread  = threading.Thread(target=gesture_video_loop, daemon=True)
        gesture_video_thread.start()

    return f"Vision mode started ({mode})."


def stop_vision():
    global ACTIVE_GESTURE, gesture_video_running, gesture_video_thread, gui_manager

    if ACTIVE_GESTURE != "vision":
        return "Vision mode is not active."

    print("Stopping Vision System...")
    gesture_video_running = False

    if gesture_video_thread and gesture_video_thread.is_alive():
        gesture_video_thread.join(timeout=1)

    live_video.stop_vision()
    ACTIVE_GESTURE = None

    if gui_manager:
        gui_manager.set_gesture_state(None)
    if window:
        try:
            window.evaluate_js("clearVideoFeed()")
        except Exception:
            pass

    print("Vision stopped cleanly.")
    return "Vision mode stopped."


def stop_active_gesture():
    global ACTIVE_GESTURE, cv2_frame_running

    if ACTIVE_GESTURE == "mouse":
        try:
            mouse.stop_mouse()
        except Exception:
            pass
        cv2_frame_running = False

    elif ACTIVE_GESTURE == "vision":
        try:
            live_video.stop_vision()
        except Exception:
            pass

    elif ACTIVE_GESTURE == "whiteboard":
        cv2_frame_running = False
        try:
            wb_game.stop()
        except Exception:
            pass

    elif ACTIVE_GESTURE == "tictactoe":
        cv2_frame_running = False
        try:
            ttt_game.stop()
        except Exception:
            pass

    ACTIVE_GESTURE = None
    if gui_manager:
        gui_manager.set_gesture_state(None)


def stop_mouse():
    global cv2_frame_running, ACTIVE_GESTURE
    if ACTIVE_GESTURE != "mouse":
        return "Virtual mouse is not active."
    cv2_frame_running = False
    mouse.stop_mouse()
    ACTIVE_GESTURE = None
    if gui_manager:
        gui_manager.set_gesture_state(None)
    return "Virtual mouse stopped."


def start_mouse():
    global cv2_frame_running, ACTIVE_GESTURE
    if ACTIVE_GESTURE == "mouse":
        return "Virtual mouse is already active."
    stop_active_gesture()
    ACTIVE_GESTURE = "mouse"
    threading.Thread(target=mouse.start_mouse, daemon=True).start()
    cv2_frame_running = True
    threading.Thread(target=cv2_frame_sender, daemon=True).start()
    return "Virtual mouse started."


def start_whiteboard():
    global cv2_frame_running, ACTIVE_GESTURE
    if ACTIVE_GESTURE == "whiteboard":
        return "Whiteboard is already active."
    stop_active_gesture()
    wb_game.start()
    ACTIVE_GESTURE    = "whiteboard"
    cv2_frame_running = True
    threading.Thread(target=cv2_frame_sender, daemon=True).start()
    return "Whiteboard started."


def stop_whiteboard():
    global cv2_frame_running, ACTIVE_GESTURE
    if ACTIVE_GESTURE != "whiteboard":
        return "Whiteboard is not active."
    cv2_frame_running = False
    wb_game.stop()
    ACTIVE_GESTURE = None
    if gui_manager:
        gui_manager.set_gesture_state(None)
    return "Whiteboard stopped."


def start_tictactoe():
    global cv2_frame_running, ACTIVE_GESTURE
    if ACTIVE_GESTURE == "tictactoe":
        return "Tic Tac Toe is already running."
    stop_active_gesture()
    ttt_game.start()
    ACTIVE_GESTURE    = "tictactoe"
    cv2_frame_running = True
    threading.Thread(target=cv2_frame_sender, daemon=True).start()
    return "Tic tac toe started."


def reset_tictactoe():
    if ACTIVE_GESTURE != "tictactoe":
        return "Tic Tac Toe is not running."
    ttt_game.reset_game()
    return "Game reset."


def stop_tictactoe():
    global cv2_frame_running, ACTIVE_GESTURE
    if ACTIVE_GESTURE != "tictactoe":
        return "Tic Tac Toe is not running."
    cv2_frame_running = False
    ttt_game.stop()
    ACTIVE_GESTURE = None
    if gui_manager:
        gui_manager.set_gesture_state(None)
    return "Game stopped."


def gesture_video_loop():
    global gesture_video_running, gui_manager

    while gesture_video_running:
        frame = None
        if ACTIVE_GESTURE == "whiteboard":
            frame = wb_game.get_frame()
        elif ACTIVE_GESTURE == "tictactoe":
            frame = ttt_game.get_frame()
        elif ACTIVE_GESTURE == "vision":
            frame = live_video.get_frame()
        if frame is not None and gui_manager:
            gui_manager.gui.update_mouse_frame(frame)
        time.sleep(0.03)

# ── Intention detection ───────────────────────────────────────────────────────

def intention_detector(text):
    global intention

    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY') or GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You predict the intention of the user and pick ONE label from this list:\n"
                    "['open application_name', 'close application_name', "
                    "'search for latest data', 'exit', 'casual conversation', "
                    "'turning on light', 'turning off light', "
                    "'press keyboard key_name', 'use virtual mouse', 'stop virtual mouse', "
                    "'use whiteboard', 'stop whiteboard', 'type', "
                    "'play tic tac toe', 'stop tic tac toe', 'reset game', "
                    "'start live video', 'stop live video', "
                    "'start share screen', 'stop share screen', "
                    "'intelligent file opener', 'research topic', 'automation mode']\n\n"
                    "Rules:\n"
                    "1. Open app → 'open application_name' (replace application_name).\n"
                    "2. Close app → 'close application_name' (replace application_name).\n"
                    "   If user wants help, not action → 'casual conversation'.\n"
                    "3. User wants to leave → 'exit'.\n"
                    "4. Query needing current data (news, prices, president, weather) → 'search for latest data'.\n"
                    "5. Turn on light → 'turning on light'.\n"
                    "6. Turn off light → 'turning off light'.\n"
                    "7. Nothing else fits → 'casual conversation'.\n"
                    "8. Press a key → 'press keyboard key_name' (replace key_name).\n"
                    "9. Use virtual mouse → 'use virtual mouse'.\n"
                    "10. Use/stop whiteboard → 'use whiteboard' or 'stop whiteboard'.\n"
                    "11. Type spoken words → 'type'.\n"
                    "12. Play tic tac toe → 'play tic tac toe'.\n"
                    "13. Stop tic tac toe → 'stop tic tac toe'.\n"
                    "14. Reset game → 'reset game'.\n"
                    "15. Start live video/vision mode → 'start live video'.\n"
                    "16. Stop live video/vision mode → 'stop live video'.\n"
                    "17. Share screen → 'start share screen'.\n"
                    "18. Stop screen share → 'stop share screen'.\n"
                    "19. Open a file, folder, or drive → 'intelligent file opener'.\n"
                    "20. Research on any topic or making PDF/ report/ notes or document → 'research topic'.\n"
                    "21. for automating tasks related to emails → 'automation mode'.\n\n"
                    "22. you need to differentiate whether the user using the words like 'search', 'open', 'close', 'report','emails', 'research', etc to make an action or just using these words to complete his casual conversation. so, seriously choose the action only when the user really wants to do it, otherwise reply with {'cmd': 'none'}."
                    "STRICT: output ONLY the label, nothing else.\n"
                    "Examples:\n"
                    "user: can you open google? → open google\n"
                    "user: how do I open google? → casual conversation\n"
                    "user: turn on the light → turning on light\n"
                    "user: adjust light brightness → casual conversation"
                     )
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            intention = response.json()["choices"][0]["message"]["content"]
            print(f"Intention detected: {intention}")
            return intention
        else:
            print(f"[intention_detector] Error {response.status_code}: {response.text[:200]}")
            return "casual conversation"
    except Exception as e:
        print(f"[intention_detector] Exception: {e}")
        return "casual conversation"

# ── Main process dispatcher ───────────────────────────────────────────────────

def process(text, intention):
    global typing_mode, last_query, SHUTDOWN_REQUESTED
    lower  = text.lower().strip()
    intent = intention.lower()

    # ── Log the user utterance to unified history ─────────────────────────────
    # (SRC_CASUAL — the main module owns this turn; other modules log their own)
    history.add("user", text, source=SRC_CASUAL)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Research cancel shortcut ──────────────────────────────────────────────
    # If research is mid-run (its _CANCELLED flag is False meaning it's active)
    # and the user says any cancel word, signal it to stop immediately.
    # This works because research.start() checks _CANCELLED between every stage.
    if research._is_cancel(lower):
        research.cancel()
        # Don't speak here — research.start() will speak the cancel message
        # when it next checks _check_cancel(). Just return so we don't also
        # hand this off to casual AI or any other handler.
        return
    # ─────────────────────────────────────────────────────────────────────────

    # ── Typing mode shortcuts (checked before intent routing) ─────────────────
    if lower.startswith("press "):
        speak(press_key(lower.replace("press ", "")))
        return
    
    if lower in ("enter typing mode", "start typing", "typing mode on", "enable typing"):
        typing_mode = True
        speak("Typing mode enabled. I will type everything you say.")
        return

    if lower in ("stop typing mode", "exit typing mode", "typing mode off", "disable typing"):
        typing_mode = False
        speak("Typing mode disabled.")
        return

    if typing_mode:
        convert_speech_to_keys(text)
        return

    if lower.startswith("type "):
        convert_speech_to_keys(lower.replace("type ", ""))
        speak("Typed.")
        return

    # ── Intent routing ────────────────────────────────────────────────────────

    if "exit" in intent:
        speak("Let today be shaped by your strength and tomorrow by your dreams. Farewell for now. Good bye.")
        is_speaking.clear()
        reflect_and_write_memory()
        shutdown_assistant()
        return

    if "start live video" in intent:
        if window:
            window.evaluate_js("activateFeature('camera', true, true)")
        speak(start_vision("camera"))
        return

    if "stop live video" in intent:
        if window:
            window.evaluate_js("activateFeature('camera', false, true)")
        speak(stop_vision())
        return

    if "play tic tac toe" in intent:
        if window:
            window.evaluate_js("activateFeature('tictactoe', true, true)")
        speak(start_tictactoe())
        return

    if "stop tic tac toe" in intent:
        if window:
            window.evaluate_js("activateFeature('tictactoe', false, true)")
        speak(stop_tictactoe())
        return

    if "reset game" in intent:
        speak(reset_tictactoe())
        return

    if "use whiteboard" in intent:
        if window:
            window.evaluate_js("activateFeature('whiteboard', true, true)")
        speak(start_whiteboard())
        return

    if "stop whiteboard" in intent:
        if window:
            window.evaluate_js("activateFeature('whiteboard', false, true)")
        speak(stop_whiteboard())
        return

    if "use virtual mouse" in intent:
        if window:
            window.evaluate_js("activateFeature('mouse', true, true)")
        speak(start_mouse())
        return

    if "stop virtual mouse" in intent:
        if window:
            window.evaluate_js("activateFeature('mouse', false, true)")
        speak(stop_mouse())
        return

    if "start share screen" in intent:
        if window:
            window.evaluate_js("activateFeature('screen', true, true)")
        speak(start_vision("screen"))
        return

    if "stop share screen" in intent:
        if window:
            window.evaluate_js("activateFeature('screen', false, true)")
        speak(stop_vision())
        return

    if "search for latest data" in intent:
        # fetch() always returns a plain string.
        # If a follow-up question is needed, it returns the question as a
        # spoken string and stores the pending query internally — the next
        # call to fetch() with the user's reply resolves it automatically.
        result = latest_data.fetch(text)
        response = clean_for_voice(str(result))
        speak(response)
        return
    
    if "turning on light" in intent:
        speak(control_light("on"))
        return

    if "turning off light" in intent:
        speak(control_light("off"))
        return

    if "intelligent file opener" in intent:
        file_opener.start(cmd=lower, speak=speak, listen=smart_listen)
        return

    if "open" in intent and "intelligent file opener" not in intent:
        speak(launch_app(intention.replace("open ", "")))
        return

    if "close" in intent:
        speak(close_app(intention.replace("close ", "")))
        return

    if "automation mode" in intent:
        automation.start(lower, speak=speak, listen=smart_listen)
        return

    if "research" in intent:
        # research.start() resolves vague topics via history.get_last_topic()
        # and logs its own turns (SRC_RESEARCH) internally
        topic_text = intent.replace("research", "").strip() or text
        research.start(topic_text, speak=speak, listen=smart_listen)
        # ── Post-research: flush audio queue so stale audio (background noise,
        #    the TTS playback echo, or anything said while PDF was building)
        #    doesn't land in the next listen() and accidentally re-trigger research.
        import time as _t
        _t.sleep(0.4)
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
            except Exception:
                break
        return

    if ACTIVE_GESTURE == "vision":
        live_video.set_query(text)
        return

    if "casual conversation" in intent:
        # ── Fallback command check ────────────────────────────────────────────
        # The intention_detector sometimes misclassifies real commands as casual
        # conversation (e.g. "launch spotify", "kill chrome", "turn on the light").
        # Before handing off to the chat AI, we run a fast second-opinion check.
        # If a real command is detected, re-route it through process() directly.
        fallback_cmd = _ai_command_check(text)
        if fallback_cmd and fallback_cmd != "none":
            print(f"[process] Fallback re-routing with intent: {fallback_cmd!r}")
            process(text, fallback_cmd)   # re-enter with the corrected intent
            return
        # ─────────────────────────────────────────────────────────────────────

        response = ai_reply(text)
        print (history)
        if not is_interrupted.is_set():
            speak(response)
        else:
            is_interrupted.clear()

# ── Audio stream teardown ─────────────────────────────────────────────────────

def close_audio_stream():
    global _persistent_stream
    if _persistent_stream:
        try:
            _persistent_stream.stop()
            _persistent_stream.close()
        except Exception:
            pass
        _persistent_stream = None

# ── Mic / GUI API ─────────────────────────────────────────────────────────────

MIC_ENABLED   = False 
is_interrupted = threading.Event()


class Api:
    def toggle_mic(self, state):
        global MIC_ENABLED
        MIC_ENABLED = state
        if not state:
            is_interrupted.clear()
            while not audio_q.empty():
                try:
                    audio_q.get_nowait()
                except Exception:
                    break
        else:
            ensure_audio_stream()

    def toggle_feature(self, name, state):
        """Called by GUI buttons. Runs in background thread so JS bridge never blocks."""
        def _run():
            if name == "camera":
                start_vision("camera") if state else stop_vision()
            elif name == "screen":
                start_vision("screen") if state else stop_vision()
            elif name == "mouse":
                start_mouse() if state else stop_mouse()
            elif name == "whiteboard":
                start_whiteboard() if state else stop_whiteboard()
            elif name == "tictactoe":
                start_tictactoe() if state else stop_tictactoe()

            gesture_to_btn = {
                "vision":     name,
                "mouse":      "mouse",
                "whiteboard": "whiteboard",
                "tictactoe":  "tictactoe",
            }
            active_btn   = gesture_to_btn.get(ACTIVE_GESTURE)
            actual_state = "true" if active_btn == name else "false"
            if window:
                try:
                    window.evaluate_js(f"syncFeatureState('{name}', {actual_state})")
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()
    def get_api_keys(self):
        env = _read_env()
        return json.dumps({
            "GROQ_API_KEY":  env.get("GROQ_API_KEY", ""),
            "NEMOTRON_API":  env.get("NEMOTRON_API", ""),
        })

    def save_api_keys(self, payload: str):
        data = json.loads(payload)
        to_save = {}
        if data.get("GROQ_API_KEY"):
            to_save["GROQ_API_KEY"] = data["GROQ_API_KEY"]
        if data.get("NEMOTRON_API"):
            to_save["NEMOTRON_API"] = data["NEMOTRON_API"]
        _write_env(to_save)
        # Reload into os.environ so the running process picks them up immediately
        for k, v in to_save.items():
            os.environ[k] = v
        return True

    def validate_api_key(self, which: str, key: str):
        try:
            if which == "groq":
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=8
                )
                return r.status_code == 200
            elif which == "nemotron":
                r = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "nvidia/llama-3.1-nemotron-70b-instruct", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=8
                )
                return r.status_code == 200
        except Exception:
            pass
        return False    

    def get_memory(self):
        try:
            if os.path.exists(ACTIVE_MEMORY_FILE):
                with open(ACTIVE_MEMORY_FILE, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            print(f"Memory read error: {e}")
        return ""

    def save_memory(self, text):
        try:
            with open(ACTIVE_MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Memory write error: {e}")

    def get_automation_config(self):
        try:
            return automation.get_automation_config()
        except Exception as e:
            print(f"Automation config read error: {e}")
            return "{}"

    def save_automation_config(self, json_str):
        try:
            automation.save_automation_config(json_str)
        except Exception as e:
            print(f"Automation config save error: {e}")

    # ── IoT / ESP32 ──────────────────────────────────────────────────────────

    def get_iot_config(self):
        try:
            if os.path.exists(IOT_CONFIG_FILE):
                with open(IOT_CONFIG_FILE, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            print(f"IoT config read error: {e}")
        return "{}"

    def save_iot_config(self, json_str):
        global ESP32_IP
        try:
            import json as _json
            os.makedirs(os.path.dirname(IOT_CONFIG_FILE), exist_ok=True)
            data = _json.loads(json_str)
            if data.get("esp32_ip"):
                ip = data["esp32_ip"].strip()
                # Auto-fix https → http (ESP32 doesn't support HTTPS)
                if ip.startswith("https://"):
                    ip = "http://" + ip[8:]
                # Auto-add http:// if missing
                if not ip.startswith("http://"):
                    ip = "http://" + ip
                data["esp32_ip"] = ip
                ESP32_IP = ip
                print(f"ESP32 IP updated to: {ESP32_IP}")
            with open(IOT_CONFIG_FILE, "w", encoding="utf-8") as f:
                _json.dump(data, f, indent=2)
        except Exception as e:
            print(f"IoT config save error: {e}")
    
    def control_light_gui(self, state):
        _load_esp32_ip()          # ← add this line so GUI buttons always use latest saved IP
        return control_light(state)

    def ping_esp32(self, ip):
        try:
            r = requests.get(f"{ip.rstrip('/')}/ping", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    # ── Text input from GUI ──────────────────────────────────────────────────

    def send_text(self, text):
        text = text.strip()
        if not text or len(text) < 2:
            return
        text_input_queue.put(text)

    def _process_text_queue(self):
        """Drain text_input_queue one entry at a time in its own thread."""
        global TEXT_MODE_ACTIVE
        while True:
            try:
                text = text_input_queue.get(timeout=1)
            except Exception:
                continue

            if not is_processing.acquire(blocking=True, timeout=10):
                continue
            try:
                TEXT_MODE_ACTIVE = True
                if window:
                    try:
                        safe = (text.replace("'", "\\'")
                                    .replace('"', '\\"')
                                    .replace("\\n", " "))
                        window.evaluate_js(f"addMessage('user', '{safe}')")
                        window.evaluate_js("updateStatus('thinking', '')")
                    except Exception:
                        pass

                if ACTIVE_GESTURE == "vision":
                    live_video.set_query(text)
                else:
                    intent = intention_detector(text)
                    process(text, intent)
            finally:
                TEXT_MODE_ACTIVE = False
                is_processing.release()

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global window

    api       = Api()
    #html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "innostaa_gui.html")
    html_path = os.path.join(BASE_DIR, "innostaa_gui.html")
    window = webview.create_window(
        "INNOSTAA",
        html_path,
        js_api=api,
        width=1100,
        height=680
    )

    def assistant_loop():
        global user_text, SHUTDOWN_REQUESTED, TEXT_MODE_ACTIVE

        greeting = random.choice(MOTIVATIONAL_GREETINGS)
        speak(greeting)

        while not SHUTDOWN_REQUESTED:
            if not MIC_ENABLED:
                if window:
                    try:
                        window.evaluate_js("updateStatus('idle', '')")
                    except Exception:
                        pass
                time.sleep(0.1)
                continue

            if window:
                try:
                    window.evaluate_js("updateStatus('listening', '')")
                except Exception:
                    pass

            live_video.USER_TYPING = True
            user_text = listen()
            live_video.USER_TYPING = False

            if not MIC_ENABLED:
                continue

            if not user_text or len(user_text.strip()) < 3:
                continue

            if is_processing.locked():
                continue

            if not is_processing.acquire(blocking=False):
                continue

            try:
                TEXT_MODE_ACTIVE = False
                if window:
                    try:
                        safe = (user_text.replace("'", "\\'")
                                         .replace('"', '\\"')
                                         .replace("\\n", " "))
                        window.evaluate_js(f"addMessage('user', '{safe}')")
                        window.evaluate_js("updateStatus('thinking', '')")
                    except Exception:
                        pass
                intent = intention_detector(user_text)
                process(user_text, intent)
            finally:
                is_processing.release()

    def background_init():
        import traceback, sys as _sys
        try:
            time.sleep(1.5)
            if window:
                window.evaluate_js("advanceSplash(10, 'Building file index...')")

            safe_print("Assistant starting - INNOSTAA is initializing...")
            file_opener.intro()

            if window:
                window.evaluate_js("advanceSplash(60, 'Preparing voice systems...')")

            time.sleep(0.3)

            import live_video as _lv
            _self = _sys.modules[__name__]   # works both as script and frozen exe

            # Share the unified history object with live_video so vision turns
            # are written into the same history the casual AI reads from.
            _lv.share_conversation_history(_self.CONVERSATION_HISTORY)
            _lv.set_window(window)
            _lv.set_speak_callback(speak)

            # ── Wire barge_in module ──────────────────────────────────────────────
            barge_in.install(
                audio_q        = audio_q,
                is_speaking    = is_speaking,
                is_interrupted = is_interrupted,
                mic_enabled_fn = lambda: MIC_ENABLED,
                window_fn      = lambda: window,
                history        = history,
                src_system     = SRC_SYSTEM,
                edge_voice     = EDGE_VOICE,
                tts_lock       = _tts_lock,
                main_module    = _self,
            )
            if window:
                window.evaluate_js("advanceSplash(85, 'Starting assistant...')")

            time.sleep(0.3)
            threading.Thread(target=api._process_text_queue, daemon=True).start()
            threading.Thread(target=assistant_loop,          daemon=True).start()

            time.sleep(0.5)
            if window:
                window.evaluate_js("splashComplete()")

        except Exception:
            log_dir = os.path.dirname(_sys.executable) if getattr(_sys, "frozen", False) \
                      else os.path.dirname(os.path.abspath(__file__))
            log_path = os.path.join(log_dir, "crash.log")
            try:
                with open(log_path, "w") as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
            try:
                if window:
                    window.evaluate_js("splashComplete()")
            except Exception:
                pass

    try:
        webview.start(
            lambda: threading.Thread(target=background_init, daemon=True).start(),
            debug=False
        )
    except Exception as e:
        print(f"Webview error: {e}")
        input("Press enter to exit")


if __name__ == "__main__":
    main()