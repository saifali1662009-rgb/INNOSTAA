import os
import time
import wave
import unicodedata
import queue
import requests
import pytesseract
import pyautogui
import threading
import mouse
from tictactoe import HandGestureTicTacToe
from whiteboard import GestureWhiteboard
from datetime import datetime, timedelta
import tempfile
import keyboard
from claude_gui import InnostaaWithGUI
from pywinauto import Desktop
from pywinauto.findwindows import ElementNotFoundError
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess
import shutil
import random
import psutil
from groq import Groq
import unicodedata
import re

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

last_query = ""
typing_mode = False
whiteboard = GestureWhiteboard()
gui_manager = None
ACTIVE_GESTURE = None
gesture_video_running = False
gesture_video_thread = None
tictactoe = HandGestureTicTacToe()

EXIT_KEYWORDS = ["bye","exit","quit","see you later","good bye","end conversation"]
UTILITY_KEYWORDS = ["weather", "time", "date", "day","capital", "define", "meaning","solve", "calculate"]

GROQ_API_KEY = "--------------------------------"  # Replace with actual API key
GROQ_MODEL = "llama-3.1-8b-instant"

PERPLEXITY_API_KEY = "--------------------------------"  # Replace with actual API key
PERPLEXITY_CALL_COUNT = 0
MAX_PERPLEXITY_CALLS = 10

PIPER_EXE = r"F:\\INNOSTAA\\piper\\piper.exe"
PIPER_MODEL = r"F:\\INNOSTAA\\piper\\models\\en_US-danny-low.onnx"
PIPER_OUTPUT = os.path.join(tempfile.gettempdir(), "innostaa_tts.wav")

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
    "Greatness begins with a single step and a powerful intention. What intention are we shaping today?",
    "Every champion was once a beginner who refused to quit. So, what challenge shall we conquer today?",
    "Your potential is massive, your ideas are powerful. Tell me, what shall we create today?",
    "Success whispers to those who dare to speak their dreams out loud. What dream are we chasing right now?",
    "A new day means a new chance to grow stronger. How can I support your journey today?",
    "Your discipline writes your destiny. So, what page are we writing today?",
    "Every small effort today becomes a big victory tomorrow. What victory are we working toward?",
    "Your mind is sharp, your goals are worthy. Tell me, what mission starts now?",
    "Greatness doesn’t happen by chance, it happens by choice. What choice are we making today?"
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

def speak(text):
    global audio_stream, gui_manager
    text = text[:300]

    print("Assistant:", text)

    if gui_manager:
        gui_manager.update_status('speaking', text, False)

    try:
        if audio_stream is not None:
            try:
                audio_stream.stop()
                audio_stream.close()
            except:
                pass
            audio_stream = None

        is_speaking.set()

        subprocess.run(
            [PIPER_EXE, "--model", PIPER_MODEL, "--output_file", PIPER_OUTPUT],
            input=text,
            text=True,
            check=True
        )

        data, fs = sf.read(PIPER_OUTPUT, dtype="float32")
        sd.play(data, fs)
        sd.wait()

    except Exception as e:
        print("Piper TTS Error:", e)

    finally:
        is_speaking.clear()

        if gui_manager:
            gui_manager.update_status('idle', '', False)

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
        # store user message in session memory
        update_memory("user", user_text)

        user_memory_text = ""
        if os.path.exists(ACTIVE_MEMORY_FILE):
            with open(ACTIVE_MEMORY_FILE, "r", encoding="utf-8") as f:
                user_memory_text = f.read().strip()

        lower_text = user_text.lower()
        allow_memory = True
        if any(word in lower_text for word in UTILITY_KEYWORDS):
            allow_memory = False

        current_date = get_current_date_str()
        current_time = get_current_time_str()
        current_year = get_current_year()    

        messages = [
            {
             "role": "system",
             "content": f"""SYSTEM CONTEXT:
                            - Today's date is {current_date}.
                            - Current time is {current_time}.       
                            - Current year is {current_year}.        
              Your name is innostaa, made by Saif for his voice assistance.
              you have ability to turn on my room light, can do casual talks, 
              can search on internet and also writes or open application when asked. 
              Respond clearly, friendly and answer briefly. you do not answer like that you are in chat,
              answer like that you are on voice conversation so do not use special characters like |,:,-, etc in conversation.
              Your knowledge is limited to 2023. you can control room light so you can ask me but never control without my permission.
              For live or current information, the system will handle it separately.
              you are made to speak only english and never aswer in other language and never understand other language.
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
              - you can remind me for any event that I told you or written in anywhere in memory but never make up anything that is not in memory.
              - keep answers concise. prefer 1-3 short sentences.
              - if more detail is useful, ask a follow-up question instead of continuing. 
              - Avoid repetitive filter phrases like " that's a great question".       
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
- Ignore casual talk, jokes, greetings
- Do NOT guess or assume
- do not removve any memory till you got any commmand to.
- rewrite the memory having old memory and new memory that you think is important to be remembered.
- Rewrite memory as clean, short paragraphs
- write only in third party.
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

        # Write new memory
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

def listen():
    global audio_stream
    safe_print("Listening...")

    START_THRESHOLD = 35      
    STOP_THRESHOLD = 25           
    MIN_SPEECH_BLOCKS = 4        
    SILENCE_TIME = 0.6           

    audio_frames = []
    speaking_started = False
    last_voice_time = time.time()  

    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except:
            break

    audio_stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback
    )
    audio_stream.start()

    while True:
        if is_speaking.is_set():
            continue

        block = audio_q.get()
        samples = np.frombuffer(block, dtype=np.int16)
        energy = int(np.abs(samples).mean())

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

    try:
        audio_stream.stop()
        audio_stream.close()
    except:
        pass

    audio_stream = None

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
    
def perplexity_search(query):
    global PERPLEXITY_CALL_COUNT

    if PERPLEXITY_CALL_COUNT >= MAX_PERPLEXITY_CALLS:
        return "Live internet search is temporarily limited."

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": """assuming that you are voice assistant. 
             behave like human not machine. 
             Answer very briefly and clearly for voice output.
            """},
            {"role": "user", "content": query}
        ]
    }

    try:
        r = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        data = r.json()
        PERPLEXITY_CALL_COUNT += 1
        return clean_for_voice(normalize_text(data["choices"][0]["message"]["content"]))

    except:
        return "I could not fetch live information right now."    

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
#import subprocess
#import shutil

def launch_app(app_name):
    """
    Improved universal launcher:
    ✔ fuzzy matches
    ✔ fixed known paths
    ✔ prevents folder mistaken as EXE
    """
    app_name = app_name.lower().strip()
    fixed_paths = {
        "edge": r"C:\\Program Files (x86)\\Microsoft\\Edge\Application\\msedge.exe",
        "microsoft edge": r"C:\\Program Files (x86)\\Microsoft\\Edge\Application\\msedge.exe",
        "chrome": r"C:\\Program Files\\Google\\Chrome\Application\\chrome.exe",
        "google chrome": r"C:\\Program Files\\Google\\Chrome\Application\\chrome.exe",
        "file explorer": r"C:\\Windows\\explorer.exe",
        "explorer": r"C:\Windows\\explorer.exe",
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
    
def uia_click(target, timeout=5):
    try:
        desktop = Desktop(backend="uia")
        active = desktop.get_active()
        element = active.child_window(title_re=f".*{target}.*")
        element.wait("exists ready", timeout=timeout)
        ctrl_type = element.element_info.control_type       
        if ctrl_type == "TabItem":
            element.select()
            return f"Opened {target} tab."
        try:
            element.invoke()
            return f"Clicked {target}."
        except:
            pass      
        try:
            element.click_input()
            return f"Clicked {target}."
        except:
            pass    
        try:
            element.set_focus()
            keyboard.press_and_release("enter")
            return f"Activated {target}."
        except:
            pass

        return f"{target} is part of a protected or web based interface and cannot be contolled."

    except ElementNotFoundError:
        return f"I could not find {target} in this window."
    except Exception:
        return "I found the window, but interaction failed."

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

ESP32_IP = "http://172.23.151.29"                                                                                    # Replace with actual IP

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
     
def stop_active_gesture():
    global ACTIVE_GESTURE

    if ACTIVE_GESTURE == "mouse":
        mouse.stop_mouse()

    elif ACTIVE_GESTURE == "whiteboard":
        stop_whiteboard()   
    elif ACTIVE_GESTURE == "tictactoe":
        stop_tictactoe()     

    ACTIVE_GESTURE = None

    if gui_manager:
        gui_manager.set_gesture_state(None)

def start_gesture_mouse():
    global mouse_thread, ACTIVE_GESTURE, gui_manager

    if ACTIVE_GESTURE == "mouse":
        return "Virtual mouse is already active."
    stop_active_gesture()

    ACTIVE_GESTURE = "mouse"

    if gui_manager:
        gui_manager.set_gesture_state("mouse")

    mouse_thread = threading.Thread(
        target=mouse.start_mouse,
        daemon=True
    )
    mouse_thread.start()

    return "Virtual mouse activated."

def stop_gesture_mouse():
    global ACTIVE_GESTURE, gui_manager

    if ACTIVE_GESTURE != "mouse":
        return "Virtual mouse is not active."

    mouse.stop_mouse()
    ACTIVE_GESTURE = None

    if gui_manager:
        gui_manager.set_gesture_state(None)

    return "Virtual mouse stopped."

def start_whiteboard():
    global whiteboard, ACTIVE_GESTURE
    global gesture_video_running, gesture_video_thread
    global gui_manager

    if ACTIVE_GESTURE == "whiteboard":
        return "Whiteboard is already active."

    stop_active_gesture()

    whiteboard.start()
    ACTIVE_GESTURE = "whiteboard"

    if gui_manager:
        gui_manager.set_gesture_state("whiteboard")

    gesture_video_running = True
    gesture_video_thread = threading.Thread(
        target=gesture_video_loop,
        daemon=True
    )
    gesture_video_thread.start()

    return "Virtual whiteboard started."

def stop_whiteboard():
    global whiteboard, ACTIVE_GESTURE
    global gesture_video_running, gui_manager

    gesture_video_running = False
    whiteboard.stop()
    ACTIVE_GESTURE = None

    if gui_manager:
        gui_manager.set_gesture_state(None)

    return "Whiteboard closed."

def start_tictactoe():
    global tictactoe, ACTIVE_GESTURE
    global gesture_video_running, gesture_video_thread, gui_manager

    if ACTIVE_GESTURE == "tictactoe":
        return "Tic Tac Toe is already running."

    stop_active_gesture()

    tictactoe.start()
    ACTIVE_GESTURE = "tictactoe"

    if gui_manager:
        gui_manager.set_gesture_state("tictactoe")

    gesture_video_running = True
    gesture_video_thread = threading.Thread(
        target=gesture_video_loop,
        daemon=True
    )
    gesture_video_thread.start()

    return "Tic Tac Toe started."

def reset_tictactoe():
    global tictactoe, ACTIVE_GESTURE

    if ACTIVE_GESTURE != "tictactoe":
        return "Tic Tac Toe is not running."

    tictactoe.reset_game()
    return "Game reset."

def stop_tictactoe():
    global tictactoe, ACTIVE_GESTURE
    global gesture_video_running, gui_manager

    gesture_video_running = False
    tictactoe.stop()
    ACTIVE_GESTURE = None

    if gui_manager:
        gui_manager.set_gesture_state(None)

    return "Tic Tac Toe closed."

def gesture_video_loop():
    global gesture_video_running, ACTIVE_GESTURE
    global whiteboard, tictactoe, gui_manager

    while gesture_video_running:
        frame = None

        if ACTIVE_GESTURE == "whiteboard":
            frame = whiteboard.get_frame()

        elif ACTIVE_GESTURE == "tictactoe":
            frame = tictactoe.get_frame()

        if frame is not None and gui_manager:
            gui_manager.gui.update_mouse_frame(frame)

        time.sleep(0.03)

def process(text):
    global typing_mode, last_query, SHUTDOWN_REQUESTED
    lower = text.lower().strip()

    if lower.startswith("press "):
        key_name = lower.replace("press ", "")
        speak(press_key(key_name))
        return
   
    if lower in ["enter typing mode", "start typing", "typing mode on", "enable typing"]:
        typing_mode = True
        speak("Typing mode enabled. I will type everything you say.")
        return

    if lower in ["stop typing mode", "exit typing mode", "typing mode off", "disable typing","switch to typing mode"]:
        typing_mode = False
        speak("Typing mode disabled.")
        return

    if typing_mode:
        convert_speech_to_keys(text)
        return  # VERY IMPORTANt

    if lower.startswith("type "):
        msg = lower.replace("type ", "")
        msg = convert_speech_to_keys(msg)
        keyboard.write(msg)
        speak("Typed.")
        return
    
    if any(word in lower for word in EXIT_KEYWORDS):
        speak("It was a pleasure talking to you. Take care.")

        is_speaking.clear()

        reflect_and_write_memory()

        shutdown_assistant()
        return    
    
    if lower in ["use virtual mouse.", "start virtual mouse", "activate virtual mouse", "turn on virtual mouse", "use virtual mouse", "start virtual mouse.", "activate virtual mouse." ]:
        speak(start_gesture_mouse())
        return
    
    if lower in ["stop virtual mouse", "disable virtual mouse","turn off virtual mouse", "stop virtual mouse.","disable virtual mouse.","turn off virtual mouse."]:
        speak(stop_gesture_mouse())
        return    
    
    if lower in ["open whiteboard", "start whiteboard", "use whiteboard.", "start whitboard.", "open whiteboard.", "use whiteboard" ]:
        speak(start_whiteboard())
        return
    
    if lower in ["close whiteboard", "stop whiteboard", "close whiteboard.", "stop whiteboard."]:
        speak(stop_whiteboard())
        return
    
    if lower in ["start game", "play game.", "play game", "start game."]:
        speak(start_tictactoe())
        return
    
    if lower in ["reset game", "restart game", "reset tic tac toe", "reset game.", "restart game."]:
       speak(reset_tictactoe())
       return
    
    if lower in ["stop game", "close game", "stop game.", "close game."]:
        speak(stop_tictactoe())
        return

    if "blank document" in lower:
        keyboard.press_and_release("enter")
        speak("Blank document opened.")
        return

    if lower.startswith("click "):
        target = lower.replace("click ", "").strip()
        response = uia_click(target)
        speak(response)
        return
       
    if "date" in lower:
        speak(f"Today is {get_current_date_str()}.")
        return
    
    if "time" in lower:
        speak(f"The current time is {get_current_time_str()}.")
        return
    
    if "year" in lower:
        speak(f"The current year is {get_current_year()}.")
        return
        
    if "new year" in lower:
        now = get_current_datetime()
        new_year = datetime(now.year + 1, 1, 1)
        days = (new_year - now).days
    
        if days <= 30:
            speak(f"New Year is coming in {days} days.")
        else:
            speak("New Year is still some time away.")
        return

    light_on_commands = ["turn on the light","switch on the light","light on", "on the light", "switch the light on", "turn the light on"]
    
    light_off_commands = ["turn off the light", "switch off the light", "light off","off the light", "turn the light off", "switch off light"  ]

    PERPLEXITY_TRIGGERS = [ "search for", "search that", "tell me", "latest", "current", "news", "weather", "who is", "who are", "what is", ]
    for trigger in PERPLEXITY_TRIGGERS:
        if lower.startswith(trigger):
            speak("Checking live information...")
            result = perplexity_search(text)
            speak(result)
            last_query = text
            return
    
    if lower.startswith("open "):
        app = lower.replace("open ", "")
        speak(launch_app(app))
        return

    if lower.startswith("close "):
        app = lower.replace("close ", "")
        speak(close_app(app))
        return
        
    for cmd in light_on_commands:
        if cmd in lower:
            response = control_light("on")
            speak(response)
            return
    
    for cmd in light_off_commands:
        if cmd in lower:
            response = control_light("off")
            speak(response)
            return

    reply = ai_reply(text)
    speak(reply)

def main():
    global gui_manager

    safe_print("Assistant starting – INNOSTAA is initializing...")
    print(PERPLEXITY_CALL_COUNT)

    gui_manager = InnostaaWithGUI()
    gui_manager.start_gui()

    mouse.set_frame_callback(gui_manager.gui.update_mouse_frame)

    def assistant_loop():
        greeting = random.choice(MOTIVATIONAL_GREETINGS)
        speak(greeting)
    
        while not SHUTDOWN_REQUESTED:
            if gui_manager:
                gui_manager.update_status('listening', 'Listening...', True)
    
            user_text = listen()   
    
            if not user_text:
                if gui_manager:
                    gui_manager.update_status('idle', '', False)
                continue
    
            if gui_manager:
                gui_manager.update_status('idle', f'You: {user_text}', False)
    
            process(user_text)

    threading.Thread(target=assistant_loop, daemon=True).start()
    gui_manager.mainloop()

if __name__ == "__main__":
    main()