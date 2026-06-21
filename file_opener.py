import os
import json
import subprocess
import platform
from openai import OpenAI
from rapidfuzz import fuzz
import dotenv

dotenv.load_dotenv()

# ── LLM setup ────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NEMOTRON_API")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "meta/llama-3.1-8b-instruct"

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

# ── Constants ─────────────────────────────────────────────────────────────────
DRIVES = ["C", "D", "E", "F", "G"]
FUZZY_THRESHOLD = 72

# User-owned folders only — never touch system paths
USER_FOLDERS = [
    os.path.join(os.path.expanduser("~"), "Documents"),
    os.path.join(os.path.expanduser("~"), "Downloads"),
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.join(os.path.expanduser("~"), "Pictures"),
    os.path.join(os.path.expanduser("~"), "Music"),
    os.path.join(os.path.expanduser("~"), "Videos"),
    os.path.join(os.path.expanduser("~"), "OneDrive"),
]

# Extra non-C drives are fully user-owned, scan them entirely
EXTRA_DRIVES = [f"{d}:\\" for d in ["D", "E", "F", "G"]]


# ── AI: extract intent from command ──────────────────────────────────────────
def ai_extract_context(command: str) -> dict:
    """Ask the LLM to parse the user's command into structured intent."""
    prompt = f"""
You are a file assistant. Extract information from the user command below.
Return ONLY a valid JSON object with these exact keys:
- "keywords": list of important filename or folder-name related words
  (exclude stop-words like open, can, you, please, the, a, file, my, show, me)
- "explicit_path": the full or partial path if the user mentioned one, else null
  (e.g. "C:\\Users\\John\\report.pdf" or "D:\\Projects")
- "drive": single drive letter like C or D or null if not mentioned
- "filetype": file extension like pdf, xlsx, py, docx or null if not mentioned
- "action": one of "open", "summarize_only", "open+summarize"
- "target_type": one of "file", "folder", "drive"

User command: "{command}"

Return only JSON. No explanation. No markdown.
"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {
            "keywords": [],
            "explicit_path": None,
            "drive": None,
            "filetype": None,
            "action": "open",
            "target_type": "file",
        }


# ── Drive helpers ─────────────────────────────────────────────────────────────
def detect_drive_command(command: str) -> str | None:
    """Quick check: does the user just want to open a drive like 'open D drive'?"""
    cmd = command.lower()
    for d in DRIVES:
        if f"{d.lower()} drive" in cmd or f"open {d.lower()}:" in cmd:
            return d.upper()
    return None


def open_drive(drive_letter: str, speak=None):
    path = f"{drive_letter}:\\"
    if os.path.exists(path):
        _launch(path)
        if speak:
            speak(f"Opening {drive_letter} drive.")
    else:
        if speak:
            speak(f"{drive_letter} drive was not found on this system.")


# ── Cross-platform launcher ───────────────────────────────────────────────────
def _launch(path: str):
    """Open a file, folder, or drive with the system's default handler."""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])


# ── Folder search (lightweight, user dirs only) ───────────────────────────────
def find_folder(keyword: str, drive_hint: str | None = None) -> str | None:
    """
    Search for a folder by fuzzy name match.
    Looks only in user-owned locations — never system paths.
    """
    if drive_hint and drive_hint.upper() != "C":
        roots = [f"{drive_hint}:\\"]
    else:
        roots = USER_FOLDERS + [d for d in EXTRA_DRIVES if os.path.exists(d)]

    for root_path in roots:
        if not os.path.exists(root_path):
            continue
        for root, dirs, _ in os.walk(root_path):
            for folder_name in dirs:
                clean = folder_name.replace("_", " ").replace("-", " ")
                if fuzz.token_sort_ratio(keyword.lower(), clean.lower()) >= FUZZY_THRESHOLD:
                    return os.path.join(root, folder_name)
    return None


# ── File search (user dirs only, no indexing) ─────────────────────────────────
def find_files(
    keyword: str,
    filetype: str | None = None,
    drive_hint: str | None = None,
    excluded: set | None = None,
) -> list[str]:
    """
    Search for files by fuzzy name match within user-owned directories only.
    Returns a list of matching full paths, best matches first.
    No index file. No system folder crawl.
    """
    if excluded is None:
        excluded = set()

    if drive_hint and drive_hint.upper() != "C":
        roots = [f"{drive_hint}:\\"]
    else:
        roots = USER_FOLDERS + [d for d in EXTRA_DRIVES if os.path.exists(d)]

    scored: list[tuple[str, int]] = []

    for root_path in roots:
        if not os.path.exists(root_path):
            continue
        for root, dirs, files in os.walk(root_path):
            # Skip hidden dirs to stay fast
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in files:
                full_path = os.path.join(root, filename)
                if full_path in excluded:
                    continue
                ext = os.path.splitext(filename)[1].replace(".", "").lower()
                if filetype and ext != filetype.lower():
                    continue
                name_clean = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                score = fuzz.token_sort_ratio(keyword.lower(), name_clean.lower())
                if score >= FUZZY_THRESHOLD:
                    scored.append((full_path, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [path for path, _ in scored]


# ── File content reader (for summarize) ──────────────────────────────────────
def read_file_content(filepath: str) -> str | None:
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in [".txt", ".py", ".js", ".cpp", ".md", ".csv", ".log"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:3000]
        elif ext == ".pdf":
            import fitz
            doc = fitz.open(filepath)
            return "".join([page.get_text() for page in doc])[:3000]
        elif ext in [".xlsx", ".xls"]:
            import openpyxl
            wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
            ws = wb.active
            rows = []
            for row in ws.iter_rows(values_only=True):
                rows.append(" | ".join([str(c) if c is not None else "" for c in row]))
                if len(rows) > 100:
                    break
            return "\n".join(rows)
        elif ext == ".docx":
            import docx
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])[:3000]
    except Exception:
        return None
    return None


def summarize_file(filepath: str, speak=None):
    content = read_file_content(filepath)
    if not content:
        msg = "I could not read this file to summarize it."
        if speak:
            speak(msg)
        else:
            print(msg)
        return
    prompt = f"Give a brief clear summary of this file content:\n\n{content}"
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    summary = response.choices[0].message.content.strip()
    if speak:
        speak(summary)
    else:
        print(summary)


# ── Confirmation helper ───────────────────────────────────────────────────────
_YES = {"yes", "yeah", "sure", "ok", "yep", "open", "do it", "go ahead"}
_NO  = {"no", "nope", "nah", "not this", "skip", "next"}
_STOP = {"stop", "cancel", "enough", "nevermind", "no more", "quit", "exit"}

def _ask_confirm(prompt_text: str, listen=None) -> str:
    """Return 'yes', 'no', or 'stop'."""
    if listen:
        raw = listen().lower().strip()
    else:
        raw = input(prompt_text + " (yes / no / stop): ").lower().strip()

    if any(w in raw for w in _STOP):
        return "stop"
    if any(w in raw for w in _YES):
        return "yes"
    return "no"


# ── Main entry point ──────────────────────────────────────────────────────────
def open_file(command: str, speak=None, listen=None):
    """
    Handle a natural-language command to open a file, folder, or drive.

    Priority order:
      1. Drive shortcut  ("open D drive")
      2. Explicit path given by user  ("open C:\\Users\\me\\report.pdf")
      3. Folder search
      4. File search (user dirs only)
    """
    command_lower = command.lower()

    # 1 ── Quick drive shortcut
    quick_drive = detect_drive_command(command)
    if quick_drive:
        open_drive(quick_drive, speak)
        return

    # 2 ── AI parses intent
    context = ai_extract_context(command)
    target_type = context.get("target_type", "file")
    action      = context.get("action", "open")
    explicit    = context.get("explicit_path")
    keyword     = " ".join(context.get("keywords", []))
    filetype    = context.get("filetype")
    drive_hint  = context.get("drive")

    # ── Drive open via AI context
    if target_type == "drive":
        drive = drive_hint
        if drive:
            open_drive(drive, speak)
        else:
            msg = "Which drive would you like to open?"
            if speak:
                speak(msg)
            else:
                print(msg)
        return

    # ── Explicit path given by user
    if explicit and os.path.exists(explicit):
        display = os.path.basename(explicit) or explicit
        answer = _ask_confirm(
            f'I found "{display}" at {explicit}. Should I open it?', listen
        )
        if speak:
            speak(f'Should I open "{display}"?')
        answer = _ask_confirm("", listen) if speak else answer

        if answer == "yes":
            _launch(explicit)
            if speak:
                speak(f"Opening {display}.")
            if action == "open+summarize" and os.path.isfile(explicit):
                summarize_file(explicit, speak)
        elif answer == "stop":
            if speak:
                speak("Alright, cancelled.")
        return

    # ── Folder search
    if target_type == "folder":
        if not keyword:
            msg = "Please tell me the folder name you are looking for."
            if speak:
                speak(msg)
            else:
                print(msg)
            return

        folder_path = find_folder(keyword, drive_hint)
        if folder_path:
            folder_name = os.path.basename(folder_path)
            if speak:
                speak(f'Found folder "{folder_name}". Should I open it?')
            answer = _ask_confirm(f'Found "{folder_name}" at {folder_path}. Open it?', listen)
            if answer == "yes":
                _launch(folder_path)
                if speak:
                    speak(f"Opening folder {folder_name}.")
            elif answer == "stop":
                if speak:
                    speak("Cancelled.")
        else:
            msg = "I could not find that folder in your personal directories."
            if speak:
                speak(msg)
            else:
                print(msg)
        return

    # ── File search
    if not keyword:
        msg = "I couldn't understand which file you want. Please be more specific."
        if speak:
            speak(msg)
        else:
            print(msg)
        return

    excluded: set[str] = set()

    while True:
        candidates = find_files(keyword, filetype=filetype, drive_hint=drive_hint, excluded=excluded)

        if not candidates:
            msg = "I couldn't find any more matching files in your personal folders."
            if speak:
                speak(msg)
            else:
                print(msg)
            return

        # Pick best candidate (first in sorted list)
        full_filepath = candidates[0]
        filename = os.path.basename(full_filepath)

        if not os.path.exists(full_filepath):
            excluded.add(full_filepath)
            continue

        # Summarize-only — no confirmation needed to open
        if action == "summarize_only":
            summarize_file(full_filepath, speak)
            return

        # Ask for confirmation
        if speak:
            speak(f'I found "{filename}". Should I open it?')
        answer = _ask_confirm(f'Found "{filename}" at {full_filepath}. Open it?', listen)

        if answer == "stop":
            if speak:
                speak("Alright, stopping the search.")
            return

        if answer == "yes":
            _launch(full_filepath)
            if speak:
                speak(f"Opening {filename}.")
            if action == "open+summarize":
                if speak:
                    speak("Give me a moment to summarize it.")
                summarize_file(full_filepath, speak)
            return

        # User said no — try next match
        excluded.add(full_filepath)
        if speak:
            speak("Let me look for another match.")
def intro():
    print("Building index...")
    #count = build_index()      
    #print(f"total file counted {count}") 

# ── Module entry points ───────────────────────────────────────────────────────
def start(cmd: str, speak=None, listen=None):
    open_file(cmd, speak=speak, listen=listen)