import os                                          # helps to use OS (writing/reading files)
import time                                          # time handling
import wave                                              # handles WAV audio files
import unicodedata                                       # for text normalization
import queue                                        # for audio data queue  
import requests                                     # for HTTP requests(perpexility search)
import pytesseract                                  # OCR
import pyautogui                                    # GUI automation
import threading                                    # for running many things at once
import mouse                                        # mouse control
#from tictactoe import HandGestureTicTacToe 
import tictactoe as ttt_module
import research
import automation
import whiteboard as wb_module       
import cv2
import base64
#from whiteboard import GestureWhiteboard
from datetime import datetime, timedelta               # date and time handling                                           # image processing                                      # Windows GUI interaction
from fuzzywuzzy import fuzz                            # compares string similarity
import keyboard                                         # keyboard automation
import pyttsx3                                          # text to speech
from claude_gui import InnostaaWithGUI
from pywinauto import Desktop                           # UI automation
from pywinauto.findwindows import ElementNotFoundError  # UI element errors
from datetime import datetime                        # helps getting current date and time
import numpy as np                                       # numerical operations(voice inoputs)
import sounddevice as sd                                 # audio input
import soundfile as sf                                   # audio file handling
import subprocess                                        # launching apps
import shutil                                            # moving files, copying memory files            
import random                                            # random choices
import psutil                                            # detects running apps
from groq import Groq                                    # provide intelligence
import re                                                # cleaning text
import live_video
import file_opener
import tempfile
import webview
from dotenv import load_env
loadenv()

_tts_engine = None
_tts_lock = threading.Lock()
window = None
SHUTDOWN_REQUESTED = False
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory")
MEMORY_VERSIONS_DIR = os.path.join(MEMORY_DIR, "versions")
ACTIVE_MEMORY_FILE = os.path.join(MEMORY_DIR, "user_memory.txt")
os.makedirs(MEMORY_VERSIONS_DIR, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
audio_stream = None
is_speaking = threading.Event()
last_typed_char = ""
mouse_thread = None
gesture_mouse_active = False
vision_thread = None
VIDEO_MODE = None
ttt_game = ttt_module.HandGestureTicTacToe()
wb_game  = wb_module.GestureWhiteboard()
cv2_frame_running = False
text_input_queue = queue.Queue()
is_processing    = threading.Lock()
TEXT_MODE_ACTIVE = False
last_query = ""
typing_mode = False
#whiteboard = GestureWhiteboard()
gui_manager = None
ACTIVE_GESTURE = None
gesture_video_running = False
gesture_video_thread = None
#tictactoe = HandGestureTicTacToe()

EXIT_KEYWORDS = ["bye","exit","quit","see you later","good bye","end conversation"]
UTILITY_KEYWORDS = ["weather", "time", "date", "day","capital", "define", "meaning","solve", "calculate"]

GROQ_API_KEY = "-----------------" # Set your Groq API key in the .env file or environment variables
GROQ_MODEL = "llama-3.1-8b-instant"

SAMPLE_RATE = 16000
BLOCKSIZE = 4000

SPEECH_THRESHOLD = 230   
SILENCE_LIMIT = 4           
MIN_SPOKEN_BLOCKS = 3       

CONVERSATION_HISTORY = []
MAX_MEMORY = 6

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

groq_client = Groq(api_key=GROQ_API_KEY)

audio_q = queue.Queue()

is_speaking = threading.Event()
is_speaking.clear()

MOTIVATIONAL_GREETINGS = [
    "Once a legend said mistakes build experience, experience builds success, and success builds the future. So tell me, what are we building today?",
    "Every champion was once a beginner who refused to quit. So, what are we not quitting today?",
    "A new day means a new chance to grow stronger. How can I support your journey today?",
    "believe you can and you are halfway there. so today, where are we moving towards? ",
    "people works better when they know what the goal is and why. so, what is you goal?", 
    "the best brains of the nation may found on the last benches of the classroom. are you that back bencher?",
    "the best way to not feel hopeless is to get up and do something. so, what are we doing today?",
    "the best source of knowledge is experience. so, what are we experiencing today?",
    "A great person said I never loose. I either win or learn. As he removed loose word from his dictionary.",
    "Today is tough, tomorrow is much tougher, but the day after tomorrow is beautiful and many people give up tomorrow evening. So, do not give up, and always move ahead."
]

def safe_print(*args):
    """
    Safe print that handles ANY characters and ANY number of arguments
    without crashing on Windows cmd.
    """
    try:
        print(*args)
    except:
        fixed = []
        for a in args:
            try:
                fixed.append(str(a))
            except:
                fixed.append(str(a).encode("ascii", "ignore").decode())
        print(" ".join(fixed))

def audio_callback(indata, frames, time_info, status):
    if status:
        pass

    if is_speaking.is_set():
        return

    audio_q.put(bytes(indata))

def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        voices = _tts_engine.getProperty('voices')
        if len(voices) > 2:
            _tts_engine.setProperty('voice', voices[2].id)
        _tts_engine.setProperty('rate', 175)
        _tts_engine.setProperty('volume', 1.0)
    return _tts_engine

def speak(text):
    global window
    print("Assistant:", text)
 
    safe_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ").replace("\r", "")
    if window:
        try:
            window.evaluate_js(f"updateStatus('speaking', '{safe_text}')")
            #window.evaluate_js(f"addMessage('assistant', '{safe_text}')")
        except Exception:
            pass
 
    is_interrupted.clear()
    is_speaking.set()
 
    def _interrupt_watcher():
        INTERRUPT_THRESHOLD = 60
        INTERRUPT_BLOCKS    = 4
        count = 0
        ensure_audio_stream()
        while is_speaking.is_set():
            try:
                block = audio_q.get(timeout=0.05)
                samples = np.frombuffer(block, dtype=np.int16)
                energy  = int(np.abs(samples).mean())
                if energy > INTERRUPT_THRESHOLD:
                    count += 1
                    if count >= INTERRUPT_BLOCKS:
                        is_interrupted.set()
                        is_speaking.clear()
                        return
                else:
                    count = 0
            except Exception:
                pass
 
    watcher = threading.Thread(target=_interrupt_watcher, daemon=True)
    watcher.start()
 
    with _tts_lock:
        engine = None
        try:
            # Fresh engine every call — most reliable on Windows COM
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if len(voices) > 2:
                engine.setProperty('voice', voices[2].id)
            engine.setProperty('rate', 175)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS Error:", e)
        finally:
            try:
                if engine:
                    engine.stop()
            except Exception:
                pass
 
    is_speaking.clear()
    watcher.join(timeout=0.3)
 
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except Exception:
            break
 
    if window:
        try:
            if MIC_ENABLED:
                window.evaluate_js("updateStatus('listening', '')")
            else:
                window.evaluate_js("updateStatus('idle', '')")
        except Exception:
            pass
 
def smart_listen():
    """Follow-up input: uses text queue if text mode, else voice."""
    if TEXT_MODE_ACTIVE:
        if window:
            try: window.evaluate_js("updateStatus('listening', '')")
            except Exception: pass
        try:
            result = text_input_queue.get(timeout=30)
            if window:
                try:
                    safe = result.replace("'","\\'" ).replace('"','\\"').replace("\n"," ")
                    window.evaluate_js(f"addMessage('user', '{safe}')")
                except Exception:
                    pass
            return result
        except Exception:
            return ""
    else:
        return listen()
            
def normalize_text(text):
    try:
        cleaned = unicodedata.normalize("NFKD", text)
        cleaned = cleaned.encode("ascii", "ignore").decode()

        replacements = {
            "−": "-", "–": "-", "—": "-",
            "·": "*", "×": "*", "÷": "/",
            "√": "sqrt ",
            "π": "pi ",
            "∞": "infinity ",
            "≤": "<=", "≥": ">=", "≠": "!=",
        }

        for k, v in replacements.items():
            cleaned = cleaned.replace(k, v)

        cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)

        cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    except Exception:
        return text\
        
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
    except:
        return None
    
def update_memory(role, content):
    """
    Stores conversation turns ONLY for this session.
    This memory is NOT permanent.
    It will be used later for reflection when user exits.
    """
    CONVERSATION_HISTORY.append({
        "role": role,
        "content": content
    })

    if len(CONVERSATION_HISTORY) > MAX_MEMORY:
        CONVERSATION_HISTORY[:] = CONVERSATION_HISTORY[-MAX_MEMORY:]    

def ai_reply(user_text):
    """
    Handles normal conversation replies.
    Does NOT store long-term memory.
    """
    try:
        update_memory("user", user_text)

        user_memory_text = ""
        if os.path.exists(ACTIVE_MEMORY_FILE):
            with open(ACTIVE_MEMORY_FILE, "r", encoding="utf-8") as f:
                user_memory_text = f.read().strip()

        lower_text = user_text.lower()
        allow_memory = True
        if any(word in lower_text for word in UTILITY_KEYWORDS):
            allow_memory = False

        current_day = get_current_day()
        current_date = get_current_date_str()
        current_time = get_current_time_str()
        current_year = get_current_year()    

        messages = [
            {
             "role": "system",
             "content": f"""SYSTEM CONTEXT:
                            - Today is {current_day}.
                            - Today's date is {current_date}.
                            - Current time is {current_time}.       
                            - Current year is {current_year}.        
              Your name is innostaa, made by Saif for his voice assistance .
              you have ability to turn on my room light but can not adjust brightness, can do casual talks, 
              can search on internet and also writes or open application when asked. also, the best feature you have is you memory system which give user and AI ability to edit and also support privacy as it always remains in users PC. 
              Respond clearly, friendly and answer briefly. you do not answer like that you are in chat,
              answer like that you are on voice conversation so do not use special characters like |,:,-, etc in conversation.
              Your knowledge is limited to 2023. you can control room light so you can ask me but never control without my permission.
              For live or current information, the system will handle it separately. if any last topic for study is written in memory, you can also suggest to continue the last topic studied if the user is talking casually.
              you are weak with calender calculations so if you do those also mention that you are weak ann user should recheck it again.
              also, you not access to internet so never speak like checking live information, checking internet, or anything else.
              you are made to speak only english and never answer in other language and never understand other language.
              if you get input in other language or something like that not look like human command, strictly ignore that and give your output very briefly otherwise i will stop using you.
              Known background information about the user:
              {user_memory_text if (allow_memory and user_memory_text) else "No background information should be used for thi query."}              
              INSTRUCTIONS:
              - Use the background information naturally if it helps.
              - Do NOT mention memory, remembering, or stored information.
              - Do NOT say phrases like "I remember" or "from memory".
              - If background info is not relevant, ignore it completely.
              - Do not describe your own origin or purpose unless explicity asked.           
              - Keep responses short, clear, and voice-friendly.    
              - you can remind me for any event that is written anywhere in memory but never make up anything that is not in memory.
              - in beetween conversation get me remind for reminder if any there in memory.
              - memory have categories the information so answer with best experience for user.
              - keep answers concise. prefer 1-3 short sentences.
              - if more detail is useful, ask only one follow-up question instead of continuing.       
             """
            }
        ]
        messages.extend(CONVERSATION_HISTORY)
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages
        )
        reply = completion.choices[0].message.content.strip()
        reply = normalize_text(reply)
        update_memory("assistant", reply)
        return reply
    except Exception as e:
        return "Sorry, I faced a small issue while responding."
             
def reflect_and_write_memory():
    """
    Runs once at session end.
    Silently reflects on conversation and rewrites user memory file.
    """
    old_memory = ""
    if os.path.exists(ACTIVE_MEMORY_FILE):
        with open(ACTIVE_MEMORY_FILE, "r", encoding="utf-8") as f:
            old_memory = f.read()

    reflection_prompt = f"""
   You are a reflective personal assistant.

Existing user memory:
{old_memory if old_memory else "No prior memory."}

Conversation transcript:
{CONVERSATION_HISTORY}

TASK:
- Extract ONLY essential, stable facts explicitly stated by the user
- Ignore casual talk, jokes, greetings.
- never write anything like "**new memories**", "**updated memories**" or similar.
- write only things related to user giving his details.
- You should categories information like where it is about interests, hobbies, personal details, important dates, preferences, routines, goals, relationships, work/study, health, reminders etc.
- ignore things related to studies only write last topic which we studied and in which subject. also; homework, studies and exam preparation should be covered in work/sdtudy category.
- never write anything that is related to shopping or money.
- Do NOT guess or assume
- do not remove any memory till you got any commmand to.
- rewrite the memory having old memory and new memory that you think is important to be remembered.
- only remove anything from old memory if you tink that anything is updated during the session.
- Rewrite memory as clean, short paragraphs.
- strictly never write something does not stated by user.
- Do NOT mention the assistant
- Do NOT mention this reflection process

Output ONLY the rewritten memory text.   
   """

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": reflection_prompt}]
        )

        new_memory = completion.choices[0].message.content.strip()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        version_file = os.path.join(
            MEMORY_VERSIONS_DIR,
            f"memory_{timestamp}.txt"
        )
        
        if old_memory.strip():
            with open(version_file, "w", encoding="utf-8") as f:
                f.write(old_memory)

        with open(ACTIVE_MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write(new_memory)

    except Exception as e:
        print("Reflection error:", e)

def shutdown_assistant():
    global SHUTDOWN_REQUESTED

    SHUTDOWN_REQUESTED = True

    try:
        if gui_manager:
            gui_manager.root.after(200, gui_manager.root.destroy)
    except:
        pass

    time.sleep(0.5)
    os._exit(0)

_persistent_stream = None

def ensure_audio_stream():
    """Open the mic stream once and keep it open for the whole session."""
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
            print("Audio stream error:", e)
    return _persistent_stream

def listen():
    global audio_stream, text 
    safe_print("Listening...")

    START_THRESHOLD = 90   
    STOP_THRESHOLD = 60          
    MIN_SPEECH_BLOCKS = 3
    SILENCE_TIME = 0.35    

    audio_frames = []
    speaking_started = False
    last_voice_time = time.time()

    # If user already interrupted the last TTS, grab those frames from the queue
    # and use them as the start of this utterance
    if is_interrupted.is_set():
        is_interrupted.clear()
        # Pull whatever the watcher already put in the queue
        while True:
            try:
                block = audio_q.get_nowait()
                audio_frames.append(block)
                speaking_started = True
                last_voice_time = time.time()
            except Exception:
                break

    # Flush stale audio (but only if we didn't just grab interruption audio)
    if not speaking_started:
        while True:
            try:
                audio_q.get_nowait()
            except Exception:
                break

    # Reuse persistent stream — do NOT open a new one every call
    ensure_audio_stream()

    while True:
        # Respect mic button — exit listen() cleanly when mic is toggled off
        if not MIC_ENABLED:
            return ""

        if is_speaking.is_set():
            time.sleep(0.01)
            continue

        try:
            block = audio_q.get(timeout=0.2)   # timeout lets MIC_ENABLED check run
        except Exception:
            continue

        samples = np.frombuffer(block, dtype=np.int16)
        energy = int(np.abs(samples).mean())

        # START OF SPEECH DETECTION
        if not speaking_started:
            if energy > START_THRESHOLD:
                speaking_started = True
                audio_frames.append(block)
                last_voice_time = time.time()
            continue

        audio_frames.append(block)

        if energy > STOP_THRESHOLD:
            last_voice_time = time.time()

        if time.time() - last_voice_time > SILENCE_TIME:
            break

    # Do NOT close stream here — keep it alive for next listen() call

    if len(audio_frames) < MIN_SPEECH_BLOCKS:
        safe_print("(Ignored noise)")
        return ""

    temp_path = os.path.join(tempfile.gettempdir(), f"ania_{time.time()}.wav")

    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(audio_frames))

    try:
        with open(temp_path, "rb") as f:
            resp = groq_client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo"
            )
        os.remove(temp_path)

        text = normalize_text(resp.text.strip())
        safe_print("You:", text)
        return text

    except Exception as e:
        safe_print("STT Error:", e)
        return "" 
#def listen():
#    global text 
#    text = str(input("USER: "))
#    return text 

def clean_for_voice(text):
    text = text.replace("**", "")

    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    text = re.sub(r"\([^)]*\)", "", text)

    text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", text)

    text = re.sub(
        r"(\d+)\s*°?\s*C\b",
        r"\1 degrees Celsius",
        text
    )

    text = re.sub(
        r"(\d+)\s*°?\s*F\b",
        r"\1 degrees Fahrenheit",
        text
    )

    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)

    text = " ".join(text.split())

    return text.strip()

def launch_app(app_name):
    """
    Improved universal launcher:
    ✔ fuzzy matches
    ✔ fixed known paths
    ✔ prevents folder mistaken as EXE
    """
    app_name = app_name.lower().strip()

    fixed_paths = {
        "ed":r"C:\\Program Files (x86)\\Microsoft\\Edge\Application\\msedge.exe",
        "edge": r"C:\\Program Files (x86)\\Microsoft\\Edge\Application\\msedge.exe",
        "microsoft edge": r"C:\\Program Files (x86)\\Microsoft\\Edge\Application\\msedge.exe",
        "chrome": r"C:\\Program Files\\Google\\Chrome\Application\\chrome.exe",
        "google chrome": r"C:\\Program Files\\Google\\Chrome\Application\\chrome.exe",
        "file explorer": r"C:\\Windows\\explorer.exe",
        "explorer": r"C:\Windows\\explorer.exe",
        "word": r"C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        "microsoft word": r"C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        "whatsapp": r"C:\\Users\\{}\AppData\\Local\WhatsApp\WhatsApp.exe".format(os.getlogin()),
        "spotify": r"C:\\Users\\{}\AppData\\Roaming\Spotify\Spotify.exe".format(os.getlogin()),
        "perplexity": r"C:\\Users\\{}\AppData\\Local\\Programs\\Perplexity\\Perplexity.exe".format(os.getlogin()),
        "vs code": r"C:\\Users\\{}\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe".format(os.getlogin()),
        "visual studio code": r"C:\\Users\\{}\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe".format(os.getlogin()),
    }

    for key in fixed_paths:
        if key in app_name:
            path = fixed_paths[key]
            if os.path.isfile(path):
                subprocess.Popen(path)
                return f"Opening {key}."
            else:
                return f"I found {key}, but installation seems missing."
    
    built_in = {
        "notepad": "notepad",
        "calculator": "calc",
        "cmd": "cmd",
        "powershell": "powershell",
        "paint": "mspaint",
        "settings": "ms-settings:",
    }
    for key, cmd in built_in.items():
        if key in app_name:
            subprocess.Popen(cmd, shell=True)
            return f"Opening {key}."
    search_paths = [
        r"C:\\Program Files",
        r"C:\\Program Files (x86)",
        r"C:\\Users\\{}\AppData\\Local".format(os.getlogin())
    ] 
    for path in search_paths:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(".exe") and app_name.replace(" ", "") in file.lower().replace(" ", ""):
                    full = os.path.join(root, file)
                    subprocess.Popen(full)
                    return f"Opening {app_name}."
    return f"I couldn't find an application named {app_name}."

def get_active_app():
    try:
        import win32gui
        import win32process
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return psutil.Process(pid).name().lower()
    except:
        return ""
    
def close_app(app_name):
    """
    Close applications safely using fuzzy process matching.
    """
    app_name = app_name.lower().strip().replace(" ", "")

    process_map = {
        "chrome": "chrome.exe",
        "googlechrome": "chrome.exe",
        "edge": "msedge.exe",
        "microsoftedge": "msedge.exe",
        "whatsapp": "WhatsApp.exe",
        "spotify": "Spotify.exe",
        "fileexplorer": "explorer.exe",
        "explorer": "explorer.exe",
        "cmd": "cmd.exe",
        "commandprompt": "cmd.exe",
        "vs": "Code.exe",
        "vscode": "Code.exe",
        "visualstudiocode": "Code.exe",
        "perplexity": "Perplexity.exe",
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
            pname = proc.info["name"].lower()
            if target_proc.lower() in pname:
                proc.terminate()
                closed_any = True
        except:
            pass

    if closed_any:
        return f"Closing {app_name}."
    else:
        return f"I couldn't find {app_name} running."
    
def press_key(key_name):
    key_map = {
        "enter": "enter",
        "enter.": "enter",
        "space": "space",
        "escape": "esc",
        "backspace": "backspace",
        "delete": "delete",
        "coma": ",",
        "qoute": "'",
        "double qoute": '"',
        "full stop": ".",
        "question mark": "?",
        "exclamation mark": "!",
        "colon": ":",
        "semicolon": ";",
        "slash": "/",
        "backslash": "\\",
        "dash": "_",
        "hyphen": "-",
        "ctrl c": ["ctrl", "c"],
        "control c": ["ctrl", "c"],
        "ctrl v": ["ctrl", "v"],
        "control v": ["ctrl", "v"],
        "ctrl x": ["ctrl", "x"],
        "control x": ["ctrl", "x"]
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
        "backspace": "backspace",
        "delete": "delete",
        "enter": "enter",
        "new line": "enter",
        "space": "space",
    }

    punctuation = {
        "full stop": ".",
        "period": ".",
        "comma": ",",
        "question mark": "?",
        "exclamation mark": "!",
        "colon": ":",
        "semicolon": ";",
    }

    text = text.lower().strip()

    if text in command_keys:
        keyboard.press_and_release(command_keys[text])
        return ""

    for word, symbol in punctuation.items():
        if word in text:
            if last_typed_char != " ":
                keyboard.write(symbol + " ")
            else:
                keyboard.write(symbol + " ")
            last_typed_char = symbol
            return ""

    keyboard.write(text + " ")
    last_typed_char = text[-1]
    return ""

ESP32_IP = "http://192.168.1.24"                                                                         # Replace with actual IP

def check_esp32():
    try:
        r = requests.get(f"{ESP32_IP}/ping", timeout=1.2)
        if r.status_code == 200:
            return True
    except:
        return False
    
def control_light(state):
     if not check_esp32():
         return "ESP32 is not connected. Please check the device."
    
     try:
         requests.get(f"{ESP32_IP}/light/{state}", timeout=1.2)
         return f"Light turned {state}."
     except:
         return "I lost connection to the ESP32."    

def media_control(action):
    """
    Controls media playback using keyboard shortcuts that work across apps.
    """
    action = action.lower().strip()
    
    actions = {
        'play': 'play/pause media',
        'pause': 'play/pause media',
        'stop': 'stop media',
        'next': 'next track',
        'next song': 'next track',
        'previous': 'previous track',
        'previous song': 'previous track',
        'volume up': 'volume up',
        'volume down': 'volume down',
        'mute': 'volume mute'
    }
    
    if action in actions:
        try:
            keyboard.press_and_release(actions[action])
            return f"Media {action} executed."
        except Exception as e:
            safe_print(f"Media control error: {e}")
            return f"Could not execute {action}."
    
    return f"Unknown media command: {action}"

def cv2_frame_sender():
    global cv2_frame_running
    while cv2_frame_running:
        frame = None
 
        if ACTIVE_GESTURE == "tictactoe" and ttt_game.running:
            frame = ttt_game.get_frame()
        elif ACTIVE_GESTURE == "whiteboard" and wb_game.running:
            frame = wb_game.get_frame()
        elif ACTIVE_GESTURE == "mouse":
            # mouse.py uses frame_callback — read from it
            frame = mouse.get_latest_frame()
 
        if frame is not None and window:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 40]
            _, buf = cv2.imencode('.jpg', frame, encode_params)
            b64 = base64.b64encode(buf).decode('utf-8')
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
    global live_video, ACTIVE_GESTURE
    global gesture_video_running, gesture_video_thread
    global gui_manager

    if ACTIVE_GESTURE == "vision":
        return "Vision mode is already active."

    # 🔥 Stop any running mode
    stop_active_gesture()

    # 🔥 Start backend with mode
    live_video.start_vision(mode)
    ACTIVE_GESTURE = "vision"

    # 🔥 Update GUI
    if gui_manager:
        gui_manager.set_gesture_state("vision")

    # 🔥 Start ONLY if not already running
    if not gesture_video_running:
        gesture_video_running = True

        gesture_video_thread = threading.Thread(
            target=gesture_video_loop,
            daemon=True
        )
        gesture_video_thread.start()

    return f"Vision mode started ({mode})."

def stop_vision():
    global live_video, ACTIVE_GESTURE
    global gesture_video_running, gesture_video_thread
    global gui_manager

    if ACTIVE_GESTURE != "vision":
        return "Vision mode is not active."

    print("🛑 Stopping Vision System...")

    # 🔥 Stop gesture loop
    gesture_video_running = False

    # 🔥 Wait for thread to finish
    if gesture_video_thread and gesture_video_thread.is_alive():
        gesture_video_thread.join(timeout=1)

    # 🔥 Stop backend
    live_video.stop_vision()

    ACTIVE_GESTURE = None

    # 🔥 Clear GUI
    if gui_manager:
        gui_manager.set_gesture_state(None)

    print("✅ Vision stopped cleanly")

    return "Vision mode stopped."
     
def stop_active_gesture():
    global ACTIVE_GESTURE, cv2_frame_running

    if ACTIVE_GESTURE == "mouse":
        try:
            mouse.stop_mouse()
        except Exception:
            pass
        cv2_frame_running = False

    elif ACTIVE_GESTURE == 'vision':
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
    mouse.start_mouse()
    ACTIVE_GESTURE = "mouse"
    cv2_frame_running = True
    threading.Thread(target=cv2_frame_sender, daemon=True).start()
    return "Virtual mouse started."

def start_whiteboard():
    global cv2_frame_running, ACTIVE_GESTURE
    if ACTIVE_GESTURE == "whiteboard":
        return "Whiteboard is already active."
    stop_active_gesture()
    wb_game.start()
    ACTIVE_GESTURE = "whiteboard"
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
#def start_whiteboard():
#    global whiteboard, ACTIVE_GESTURE
#    global gesture_video_running, gesture_video_thread
#    global gui_manager
#
#    if ACTIVE_GESTURE == "whiteboard":
#        return "Whiteboard is already active."
#
#    stop_active_gesture()
#
#    whiteboard.start()
#    ACTIVE_GESTURE = "whiteboard"
#
#    if gui_manager:
#        gui_manager.set_gesture_state("whiteboard")
#
#    gesture_video_running = True
#    gesture_video_thread = threading.Thread(
#        target=gesture_video_loop,
#        daemon=True
#    )
#    gesture_video_thread.start()
#
#    return "Virtual whiteboard started."
#
#def stop_whiteboard():
#    global whiteboard, ACTIVE_GESTURE
#    global gesture_video_running, gui_manager
#
#    gesture_video_running = False
#    whiteboard.stop()
#    ACTIVE_GESTURE = None
#
#    if gui_manager:
#        gui_manager.set_gesture_state(None)
#
#    return "Whiteboard closed."

def start_tictactoe():
    global cv2_frame_running, ACTIVE_GESTURE
    if ACTIVE_GESTURE == "tictactoe":
        return "Tic Tac Toe is already running."
    stop_active_gesture()
    ttt_game.start()
    ACTIVE_GESTURE = "tictactoe"
    cv2_frame_running = True
    threading.Thread(target=cv2_frame_sender, daemon=True).start()
    return "Tic tac toe started."

#def start_tictactoe():
#    global tictactoe, ACTIVE_GESTURE
#    global gesture_video_running, gesture_video_thread, gui_manager
#
#    if ACTIVE_GESTURE == "tictactoe":
#        return "Tic Tac Toe is already running."
#
#    stop_active_gesture()
#
#    tictactoe.start()
#    ACTIVE_GESTURE = "tictactoe"
#
#    if gui_manager:
#        gui_manager.set_gesture_state("tictactoe")
#
#    gesture_video_running = True
#    gesture_video_thread = threading.Thread(
#        target=gesture_video_loop,
#        daemon=True
#    )
#    gesture_video_thread.start()
#
#    return "Tic Tac Toe started. best of luck for the game"

def reset_tictactoe():
    global tictactoe, ACTIVE_GESTURE

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
#def stop_tictactoe():
#    global tictactoe, ACTIVE_GESTURE
#    global gesture_video_running, gui_manager
#
#    gesture_video_running = False
#    tictactoe.stop()
#    ACTIVE_GESTURE = None
#
#    if gui_manager:
#        gui_manager.set_gesture_state(None)
#
#    return "Tic Tac Toe closed."

def gesture_video_loop():
    global gesture_video_running, ACTIVE_GESTURE
    global whiteboard, tictactoe, gui_manager

    while gesture_video_running:
        frame = None

        if ACTIVE_GESTURE == "whiteboard":
            frame = wb_module.get_frame()

        elif ACTIVE_GESTURE == "tictactoe":
            frame = ttt_module.get_frame()

        elif ACTIVE_GESTURE =='vision':
            frame = live_video.get_frame()    

        if frame is not None and gui_manager:
            gui_manager.gui.update_mouse_frame(frame)

        time.sleep(0.03)

def intention_detector(text):
        global intention 

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": """you have given task to predict the intention of the user and have to pick up the word from the list which best describes the intention of the user. 
                                                the list is ['open application_name', 'close application_name', 'search for latest news' 'exit', 'casual conversation', 'turning on light', ' turning off light', ' press keyboard key_name', 'use virtual mouse', 'stop virtual mouse', 'use whiteboard', 'stop whiteboard', 'type', ' play tic tac toe', 'stop tic tac toe', 'reset game', 'start live video', 'stop live video', 'start share screen', 'stop share screen', 'intelligent file opener', 'research topic', 'automation mode' ].
                                                rules to follow:
                                                1. if the user is asking to open any application then the intention will be 'open application_name' and replace the application_name with the application user wants to open.
                                                2. if the user is asking to close any application then the intention will be 'close application_name' and replace the application_name with the application user wants to close.
                                                    note- you have to differentiate that do the user wants to open or close application or just willing to get help if he is willing for help then the intention should be 'casual conversation'
                                                3. if the user wants to leave then the intention will be 'exit'.
                                                4. if the users query should be up to date for example he ask for president name, news, whether reports etc. the intention should be 'search for latest news'.
                                                5. if the user wants to turn on the light then the intention will be 'turning on light'.
                                                6. if the user wants to turn off the light then the intention will be 'turning off light'.
                                                7. if nothing fits in the above rules the intention of the user should be 'casual conversation'
                                                8. if the user is asking to press any key on keyboard then the intention will be 'press keyboard key_name' and replace the key_name with the key user wants to press.
                                                9. if the user is asking to use virtual mouse then the intention will be 'use virtual mouse'.
                                                10. if the user is asking to use or stop whiteboard then the intention will be 'use whiteboard' or 'stop whiteboard' repectively.
                                                11. if the user is asking to type that he speaks then the intention will be 'type'.
                                                12. if the user asks for playing in-built game or tic tac toe game, then the intention should be 'play tic tac toe'.
                                                13. if the user asks to stop the in-built game or tic tac toe game anything else, then the intention should be 'stop tic tac toe'.
                                                14. if the user asks to reset the tic tac toe game or in-built game, then the intention should be ' reset game'.
                                                15. if the user asks to start live video or live video call with AI or you or if user wants to use vision mode, then the intention should be 'start live video'.
                                                16. if the user asks to stop live video or live video or vision mode with AI, the the intention should be 'stop live video'.
                                                17. if the user wants to share screen with you, then the intention should be 'start share screen'.
                                                18. if the user wants to stop the screen sharing then the intention should be 'stop share screen'.
                                                19. if the user wants to open any file, folder or drive. the intention should be 'intelligent file opener'.
                                                20. if the user wants to perform research on a specific topic, then the intention should be 'research topic' and replace the topic with the specific topic the user is interested in.
                                                21. if the user wants to automate tasks like sending emails, organising files or other, then the intention should be 'automation mode'.
                 strict warning: you have to follow the rules and never output anything extra than the list of intentions i have provided to you.
                 i am giving the examples which will help you, you have just idea the command can be similar like these:
                       user: can you open google?
                       you : open google
                       user: can you guide me how to open google?
                       you : casual conversation
                       user: can you adjust brightness of the light?
                       you : casual conversation
                       user: can you turn on the light?
                       you : turning on light"""},
                {"role": "user", "content": text}
            ],
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            intention = data["choices"][0]["message"]["content"]
            print(f"Intention detected: {intention}")
            return intention
        else:
            print("Error:", response.status_code, response.text)      

def process(text, intention):
    global typing_mode, last_query, SHUTDOWN_REQUESTED
    lower = text.lower().strip()
    intent = intention.lower()

    if lower.startswith("press "):
        key_name = lower.replace("press ", "")
        speak(press_key(key_name))
        return
   
    if lower in ["enter typing mode", "start typing", "typing mode on", "enable typing"]:
        typing_mode = True
        speak("Typing mode enabled. I will type everything you say.")
        return

    if lower in ["stop typing mode", "exit typing mode", "typing mode off", "disable typing"]:
        typing_mode = False
        speak("Typing mode disabled.")
        return

    if typing_mode:
        convert_speech_to_keys(text)
        return  

    if lower.startswith("type "):
        msg = lower.replace("type ", "")
        msg = convert_speech_to_keys(msg)
        keyboard.write(msg)
        speak("Typed.")
        return
    
    if 'exit' in intent:
        speak ("Let today be shaped by your strength and tomorrow by your dreams. Farewell for now. Good byee... ")
        is_speaking.clear()
        reflect_and_write_memory()
        shutdown_assistant()
        return  

    if 'start live video' in intent:
        if window:
            window.evaluate_js("activateFeature('camera', true, true)")
        speak(start_vision("camera"))
        return
    if 'stop live video' in intent:
        if window:
            window.evaluate_js("activateFeature('camera', false, true)")
        speak(stop_vision())
        return 
            
    if 'play tic tac toe' in intent:
        if window:
            window.evaluate_js("activateFeature('tictactoe', true, true)")
        speak(start_tictactoe())
        return
    if 'stop tic tac toe' in intent:
        if window:
            window.evaluate_js("activateFeature('tictactoe', false, true)")
        speak(stop_tictactoe())
        return
    if 'reset game' in intent:
        speak(reset_tictactoe())
        return

    if 'use whiteboard' in intent:
        if window:
            window.evaluate_js("activateFeature('whiteboard', true, true)")
        speak(start_whiteboard())
        return
    if 'stop whiteboard' in intent:
        if window:
            window.evaluate_js("activateFeature('whiteboard', false, true)")
        speak(stop_whiteboard())
        return
        
    if 'use virtual mouse' in intent:
        if window:
            window.evaluate_js("activateFeature('mouse', true, true)")
        speak(start_mouse())
        return
    if 'stop virtual mouse' in intent:
        if window:
            window.evaluate_js("activateFeature('mouse', false, true)")
        speak(stop_mouse())
        return

    if 'turning on light' in intent:
        response = control_light('on')
        speak(response)
    if 'turning off light' in intent:
        response = control_light('off')
        speak(response)

    if 'intelligent file opener' in intent:
        user_text = text.lower()
        file_opener.start(cmd=user_text, speak=speak, listen=smart_listen)       

    if 'open' in intent and 'intelligent file opener' not in intent:
        app = intention.replace("open ", "")
        speak(launch_app(app))
        return    
    if 'close' in intent:
        app = intention.replace("close ", "")
        speak(close_app(app))
        return
    
    if 'start share screen' in intent:
        if window:
            window.evaluate_js("activateFeature('screen', true, true)")
        speak(start_vision("screen"))    
        return
    if 'stop share screen' in intent:
        if window:
            window.evaluate_js("activateFeature('screen', false, true)")
        speak(stop_vision())    
        return        
    
    if 'automation mode' in intent:
        user_text = text.lower()
        automation.start(user_text, speak=speak, listen=smart_listen)
        return
        
    if 'research' in intent:
        user_text = intent.replace("research", "") 
        research.start(user_text, speak=speak, listen=smart_listen)
        return
    
    if ACTIVE_GESTURE == 'vision':
        live_video.set_query(text)
        return
    
    if 'casual conversation' in intent:
        response = ai_reply(text)
        if not is_interrupted.is_set():
            speak(response)
        else:
            is_interrupted.clear()
            return
    #if 'casual conversation' in intent:
    #    reply = ai_reply(text)
    #    speak(reply)

MIC_ENABLED = True
is_interrupted = threading.Event()   # set when user speaks during TTS
class Api:
    def toggle_mic(self, state):
        global MIC_ENABLED
        MIC_ENABLED = state
 
    def toggle_feature(self, name, state):
        # Feature functions handle ACTIVE_GESTURE themselves
        # Just call them — don't override ACTIVE_GESTURE here
        if name == 'camera':
            if state: start_vision("camera")
            else: stop_vision()
        elif name == 'screen':
            if state: start_vision("screen")
            else: stop_vision()
        elif name == 'mouse':
            if state: start_mouse()
            else: stop_mouse()
        elif name == 'whiteboard':
            if state: start_whiteboard()
            else: stop_whiteboard()
        elif name == 'tictactoe':
            if state: start_tictactoe()
            else: stop_tictactoe()
        # Sync GUI button to reflect actual state
        actual_state = "true" if (ACTIVE_GESTURE == name) else "false"
        if window:
            window.evaluate_js(f"activateFeature('{name}', {actual_state}, true)")
 
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
 
    # Text input from GUI
    def send_text(self, text):
        text = text.strip()
        if not text or len(text) < 2:
            return
        # Always put in queue first
        # If a module follow-up is waiting, smart_listen() picks it up
        text_input_queue.put(text)
 
    def _process_text_queue(self):
        # Runs in its own thread, drains text_input_queue one at a time
        global TEXT_MODE_ACTIVE
        while True:
            try:
                text = text_input_queue.get(timeout=1)
            except Exception:
                continue
 
            # Skip if voice is already processing
            if not is_processing.acquire(blocking=True, timeout=10):
                continue
            try:
                TEXT_MODE_ACTIVE = True
                if window:
                    try:
                        safe = text.replace("'","\\'").replace('"','\\"').replace("\\n"," ")
                        window.evaluate_js(f"addMessage('user', '{safe}')")
                        window.evaluate_js("updateStatus('thinking', '')")
                    except Exception:
                        pass
 
                # If vision is active, skip intent classification entirely —
                # send straight to the vision module so text queries work too.
                if ACTIVE_GESTURE == 'vision':
                    live_video.set_query(text)
                else:
                    intent = intention_detector(text)
                    process(text, intent)
            finally:
                TEXT_MODE_ACTIVE = False
                is_processing.release()
            #try:
            #    TEXT_MODE_ACTIVE = True
            #    if window:
            #        try:
            #            safe = text.replace("'","\\'").replace('"','\\"').replace("\\n"," ")
            #            window.evaluate_js(f"addMessage('user', '{safe}')")
            #            window.evaluate_js("updateStatus('thinking', '')")
            #        except Exception:
            #            pass
            #    intent = intention_detector(text)
            #    process(text, intent)
            #finally:
            #    TEXT_MODE_ACTIVE = False
            #    is_processing.release()

def main():
    global window
 
    api = Api()
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'innostaa_gui.html')
 
    window = webview.create_window(
        'INNOSTAA',
        html_path,
        js_api=api,
        width=1100,
        height=680
    )
 
    def assistant_loop():
        global user_text, SHUTDOWN_REQUESTED, TEXT_MODE_ACTIVE
 
        greeting = random.choice(MOTIVATIONAL_GREETINGS)
        safe_g = greeting.replace("'", "\\'").replace('"', '\\"').replace("\\n", " ")
        if window:
            try: window.evaluate_js(f"addMessage('assistant', '{safe_g}')")
            except Exception: pass
        speak(greeting)
 
        while not SHUTDOWN_REQUESTED:
 
            if not MIC_ENABLED:
                if window:
                    try: window.evaluate_js("updateStatus('idle', '')")
                    except Exception: pass
                time.sleep(0.1)
                continue
 
            if window:
                try: window.evaluate_js("updateStatus('listening', '')")
                except Exception: pass
 
            live_video.USER_TYPING = True
            user_text = listen()
            live_video.USER_TYPING = False
 
            if not MIC_ENABLED:
                continue
 
            if not user_text or len(user_text.strip()) < 3:
                continue
 
            # Don't process voice if text is being processed
            if is_processing.locked():
                continue
 
            if not is_processing.acquire(blocking=False):
                continue
 
            try:
                TEXT_MODE_ACTIVE = False
                if window:
                    try:
                        safe = user_text.replace("'", "\\'").replace('"', '\\"').replace("\\n", " ")
                        window.evaluate_js(f"addMessage('user', '{safe}')")
                        window.evaluate_js("updateStatus('thinking', '')")
                    except Exception:
                        pass
                intent = intention_detector(user_text)
                process(user_text, intent)
            finally:
                is_processing.release()
 
    def background_init():
        time.sleep(1.5)
 
        if window:
            window.evaluate_js("advanceSplash(10, 'Building file index...')")
 
        safe_print("Assistant starting - INNOSTAA is initializing...")
        file_opener.intro()
 
        if window:
            window.evaluate_js("advanceSplash(60, 'Preparing voice systems...')")
 
        time.sleep(0.3)
        live_video.set_window(window)
        live_video.set_speak_callback(speak)
 
        if window:
            window.evaluate_js("advanceSplash(85, 'Starting assistant...')")
 
        time.sleep(0.3)
        api_instance = api  # api is already defined in main()
        threading.Thread(target=api_instance._process_text_queue, daemon=True).start() 
        threading.Thread(target=assistant_loop, daemon=True).start()
 
        time.sleep(0.5)
        if window:
            window.evaluate_js("splashComplete()")
 
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
