"""
live_video.py  –  Vision module for INNOSTAA
─────────────────────────────────────────────
Fixes in this version
  • Both models share history bidirectionally:
      - Vision queries/replies  → injected into CONVERSATION_HISTORY
        so the casual (Groq) model always has context.
      - Casual conversation turns → pulled from CONVERSATION_HISTORY
        and included in every vision API call so the vision model
        also knows what was said before the camera was opened.
  • Double subtitle bug fixed:
      - _ai_thread NO LONGER calls addMessage() for the user bubble;
        the caller (_process_text_queue / assistant_loop in the main app)
        already does that before set_query() is called.
      - _deliver_reply() NO LONGER calls addMessage('assistant'); it only
        updates the status bar.  The speak_callback (speak() in main app)
        already calls addMessage('assistant') via its own window.evaluate_js.
        If speak_callback is absent we fall back to adding the bubble here.
  • All other improvements from the previous rewrite are retained.
"""

import base64
import traceback
import cv2
import mss
import numpy as np
import os
import requests
import threading
from conversation_history import history, SRC_VISION 
import time
from openai import OpenAI
import dotenv
from queue import Queue, Empty

dotenv.load_dotenv()

# ─────────────────────────────────────────────
#  Module-level references injected by main app
# ─────────────────────────────────────────────
window         = None   # pywebview window handle
speak_callback = None   # callable(text) → None

# Reference to the SHARED conversation history list owned by innostaa_pyttsx3.
# Set via share_conversation_history() at startup.
_shared_history      = None
_shared_history_lock = threading.Lock()

def set_window(w):
    global window
    window = w

def set_speak_callback(cb):
    global speak_callback
    speak_callback = cb

def share_conversation_history(history_list, lock=None):
    """
    Pass a reference to innostaa_pyttsx3.CONVERSATION_HISTORY (the list
    itself, not a copy).  Both models will read from and write to it.

    Call once in background_init() after the set_window / set_speak_callback
    calls:
        import live_video
        import innostaa_pyttsx3 as _main_app
        live_video.share_conversation_history(_main_app.CONVERSATION_HISTORY)
    """
    global _shared_history, _shared_history_lock
    _shared_history = history_list
    if lock is not None:
        _shared_history_lock = lock

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
API_KEY  = os.getenv("NEMOTRON_API")
API_URL  = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL    = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"#"meta/llama-3.2-90b-vision-instruct"
SOURCE = "camera"   # "camera" | "screen"

CAPTURE_FPS         = 15    # target capture rate
PREVIEW_EVERY_N     = 5     # push every Nth frame to GUI preview
HASH_DIFF_THRESHOLD = 8     # perceptual-hash distance; lower = more sensitive

# How many shared history turns to give the vision model for context
MAX_CASUAL_CONTEXT  = 3
# How many local vision turns to keep and replay
MAX_VISION_TURNS    = 3

# ─────────────────────────────────────────────
#  Shared state
# ─────────────────────────────────────────────
_running        = threading.Event()
_latest_frame   = None
_frame_lock     = threading.Lock()
_last_sent_hash = None
_vision_history = []                  # local vision-only turns
_query_queue    = Queue(maxsize=3)
_last_query_str = ""

# Set True by main app while the speech recogniser is actively listening
MIC_BUSY = False

# ─────────────────────────────────────────────
#  Public helpers
# ─────────────────────────────────────────────
def set_query(q: str):
    """
    Enqueue a user question for the vision AI.
    NOTE: the caller is responsible for showing the user bubble in the GUI
    BEFORE calling set_query().  This module will NOT add it again.
    """
    global _last_query_str
    q = q.strip()
    if not q:
        return
    if q == _last_query_str:
        return
    _last_query_str = q
    try:
        _query_queue.put_nowait(q)
    except Exception:
        try:
            _query_queue.get_nowait()
        except Empty:
            pass
        try:
            _query_queue.put_nowait(q)
        except Exception:
            pass


def get_frame():
    with _frame_lock:
        if _latest_frame is not None:
            return _latest_frame.copy()
    return None


def is_running() -> bool:
    return _running.is_set()


# ─────────────────────────────────────────────
#  Perceptual hash helpers
# ─────────────────────────────────────────────
def _phash(frame, hash_size=8) -> np.ndarray:
    small = cv2.resize(frame, (hash_size + 1, hash_size))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray[:, 1:] > gray[:, :-1]


def _hash_distance(h1, h2) -> int:
    return int(np.sum(h1 != h2))


def _frame_has_changed(frame) -> bool:
    global _last_sent_hash
    h = _phash(frame)
    if _last_sent_hash is None:
        _last_sent_hash = h
        return True
    if _hash_distance(h, _last_sent_hash) >= HASH_DIFF_THRESHOLD:
        _last_sent_hash = h
        return True
    return False


# ─────────────────────────────────────────────
#  GUI helpers  (fire-and-forget, never block)
# ─────────────────────────────────────────────
def _gui(js: str):
    if window:
        try:
            window.evaluate_js(js)
        except Exception:
            pass


def _escape(text: str) -> str:
    return (text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace("\n", " ")
            .replace("\r", ""))


def _push_preview(frame):
    try:
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
        b64 = base64.b64encode(buf).decode('utf-8')
        _gui(f"updateVideoFrame('{b64}')")
    except Exception:
        pass


# ─────────────────────────────────────────────
#  Reply delivery
# ─────────────────────────────────────────────
def _deliver_reply(reply: str):
    """
    Send the vision AI reply to the user.

    GUI bubble responsibility:
      • speak_callback  IS set  → speak() in innostaa_pyttsx3.py already calls
        addMessage('assistant', …) internally, so we only update the status bar.
      • speak_callback  NOT set → we're running standalone/test, so we add the
        bubble ourselves as a fallback.

    Either way we NEVER call addMessage() twice for the same turn.
    """
    safe = _escape(reply)

    # Status bar update (always)
    _gui(f"updateStatus('speaking', '{safe}')")

    if speak_callback:
        # speak() handles addMessage('assistant') + TTS
        speak_callback(reply)
    else:
        # Fallback: standalone mode — add bubble + print
        _gui(f"addMessage('assistant', '{safe}')")
        print("Vision AI:", reply)

    # ── Inject into shared history so casual model stays in sync ──────────
    if _shared_history is not None:
        with _shared_history_lock:
            _shared_history.append({
                "role": "assistant",
                "content": f"[Vision] {reply}"
            })
            if len(_shared_history) > 20:
                del _shared_history[:-20]
    history.add("assistant", reply, source=SRC_VISION)            


def _inject_user_to_shared(query: str):
    """Mirror the vision query into shared history for the casual model."""
    if _shared_history is not None:
        with _shared_history_lock:
                _shared_history.append({"role": "user", "content": f"[Vision query] {query}"})
        history.add("user", query, source=SRC_VISION)


# ─────────────────────────────────────────────
#  Capture thread
# ─────────────────────────────────────────────
def _capture_thread():
    global _latest_frame

    cap = sct = monitor = None
    frame_counter = 0
    sleep_time = 1.0 / CAPTURE_FPS

    if SOURCE == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Vision] Could not open camera (index 0).")
            _running.clear()
            return
    else:
        sct     = mss.mss()
        monitor = sct.monitors[1]

    print(f"[Vision] Capture thread started – source: {SOURCE}")
    import traceback
    while _running.is_set():
        try:
            if SOURCE == "camera":
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
            else:
                screenshot = sct.grab(monitor)
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

            with _frame_lock:
                _latest_frame = frame

            frame_counter += 1
            if frame_counter % PREVIEW_EVERY_N == 0:
                _push_preview(frame)
        except Exception as e:
            print(f"[Vision] Capture error: {e}")
            traceback.print_exc()

        time.sleep(sleep_time)

    if cap:
        cap.release()
    if sct:
        sct.close()
    print("[Vision] Capture thread stopped.")


# ─────────────────────────────────────────────
#  Message builder  – bidirectional context
# ─────────────────────────────────────────────
def _build_messages(user_query: str, img_b64: str) -> list:
    global system
    """
    Assemble the full message list for the vision API call:
      1. System prompt (with instruction on how to read casual-chat context)
      2. Recent casual-chat turns from shared history (text-only, for context)
      3. Recent local vision turns (text-only)
      4. Current user query + current frame
    """
    system = {
        "role": "system",
        "content": (
            "You are a helpful voice assistant with vision capabilities. "
            "Answer only what the user asked — keep replies to 2-3 sentences "
            "unless asked to elaborate. Sound natural, like a human. "
            "You may receive prior conversation turns. Turns prefixed with "
            "[Vision] or [Vision query] are from previous camera/screen sessions. "
            "Other turns are from the user's casual conversation with a text AI. "
            "Use all of them naturally as context, but never mention them explicitly "
            "or tell the user you have a conversation history."
            "never repeat your answer unless asked to. never speak same words, speak like human"
            "if you see any GUI of application named INNOSTAA, actually thats the graphical interface is of yours."
            "if user ask for tasks like writing mails, making pdg or report, opening files or applications, or anything else.. simply say 'stop using the vision mode to do this task and ask me again' and never say you can't do that, just say that to make user stop using vision mode for that task and ask again without vision context. "
        )
    }

    context_turns = []

    # ── Pull recent casual-chat turns from shared history ─────────────────
    context_turns = history.context(n=MAX_CASUAL_CONTEXT*2, exclude=[])
    context_turns=[t for t in context_turns
                   if not t["content"].startswith("Vision")]

    # ── Append local vision history ────────────────────────────────────────
    context_turns.extend(_vision_history[-(MAX_VISION_TURNS * 2):])

    # ── Current turn with image ────────────────────────────────────────────
    current_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            }
        ]
    }

    return [system] + context_turns + [current_turn]


def _update_vision_history(user_query: str, assistant_reply: str):
    _vision_history.append({"role": "user",      "content": user_query})
    _vision_history.append({"role": "assistant", "content": assistant_reply})
    max_entries = MAX_VISION_TURNS * 2
    if len(_vision_history) > max_entries:
        del _vision_history[:-max_entries]


# ─────────────────────────────────────────────
#  AI thread
# ─────────────────────────────────────────────
def _ai_thread():
    print("[Vision] AI thread started.")

    while _running.is_set():
        # 1. Wait for a query
        try:
            user_query = _query_queue.get(timeout=0.2)
        except Empty:
            continue

        # 2. Hold if mic is active
        wait_start = time.time()
        while MIC_BUSY and _running.is_set():
            if time.time() - wait_start > 5:
                break
            time.sleep(0.1)

        if not _running.is_set():
            break

        # 3. Grab frame
        frame = get_frame()
        if frame is None:
            _deliver_reply("I can't see anything yet — please wait a moment.")
            continue

        # 4. Status bar → thinking
        #    NOTE: we do NOT call addMessage('user') here.
        #    The main app already did that before calling set_query().
        _gui("updateStatus('thinking', '')")

        # 5. Frame change check (log only; always send on explicit query)
        if not _frame_has_changed(frame):
            print("[Vision] Frame similar to last sent — sending anyway.")

        # 6. Encode frame
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            _deliver_reply("I couldn't process the current frame. Please try again.")
            continue
        img_b64 = base64.b64encode(buf).decode('utf-8')

        # Show analysed frame in preview
        _gui(f"updateVideoFrame('{img_b64}')")

        # 7. Mirror query to shared history (for casual model)
        _inject_user_to_shared(user_query)

        # 8. Call vision API
        payload = {
            "model":    MODEL,
            "messages": _build_messages(user_query, img_b64),
            "max_tokens": 300,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type":  "application/json"
        }

        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=25
            )
            if response.status_code == 200:
                #reply = response.json()["choices"][0]["message"]["content"].strip()
                data = response.json()
                
                print("\n========== FULL NVIDIA RESPONSE ==========")
                import json
                print("==========================================\n")
                
                message = data["choices"][0]["message"]
                
                content = message.get("content")
                reasoning = message.get("reasoning_content")
                
                if content:
                    reply = str(content).strip()
                
                elif reasoning:
                    print("[Vision] Falling back to reasoning_content")
                
                    import re
                
                    match = re.search(
                        r"Let's craft:\s*\"([^\"]+)\"",
                        reasoning,
                        re.DOTALL
                    )
                
                    if match:
                        reply = match.group(1).strip()
                    else:
                        reply = reasoning[-500:].strip()
                
                else:
                    _deliver_reply("Model returned an empty response.")
                    continue
                _update_vision_history(user_query, reply)
                _deliver_reply(reply)
            else:
                print(f"[Vision] API error {response.status_code}: {response.text[:200]}")
                _deliver_reply("Sorry, I had trouble reaching the vision API.")

        except requests.exceptions.Timeout:
            _deliver_reply("The vision API took too long to respond. Please try again.")
        except Exception as e:
            import traceback
            print(f"[Vision] Unexpected error: {e}")
            traceback.print_exc()
            _deliver_reply("Something went wrong while processing your request.")

    print("[Vision] AI thread stopped.")


# ─────────────────────────────────────────────
#  Start / Stop
# ─────────────────────────────────────────────
def start_vision(mode: str = "camera"):
    global SOURCE, _latest_frame, _last_sent_hash

    if _running.is_set():
        if SOURCE == mode:
            print(f"[Vision] Already running in '{mode}' mode.")
            return
        stop_vision()
        time.sleep(0.4)

    SOURCE          = mode
    _latest_frame   = None
    _last_sent_hash = None

    while not _query_queue.empty():
        try:
            _query_queue.get_nowait()
        except Empty:
            break

    _running.set()

    threading.Thread(target=_capture_thread, daemon=True,
                     name="vision-capture").start()
    threading.Thread(target=_ai_thread,      daemon=True,
                     name="vision-ai").start()

    print(f"[Vision] Started — source: {mode}")


def stop_vision():
    global _latest_frame, _last_sent_hash

    if not _running.is_set():
        return

    print("[Vision] Stopping...")
    _running.clear()
    time.sleep(0.6)

    _latest_frame   = None
    _last_sent_hash = None
    _vision_history.clear()

    print("[Vision] Stopped.")