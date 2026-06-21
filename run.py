import os, sys
import setup_window

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
ENV_FILE = os.path.join(BASE_DIR, ".env")

REQUIRED_KEY_IDS = ["GROQ_API_KEY", "NEMOTRON_API"]

def load_env():
    result = {}
    if not os.path.exists(ENV_FILE): 
        return result
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: 
                continue
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result

def apply_env(env):
    for k, v in env.items():
        if v: 
            os.environ[k] = v

def keys_complete(env):
    return all(env.get(k, "").strip() for k in REQUIRED_KEY_IDS)

def main():
    # Load whatever is already saved
    env = load_env()
    apply_env(env)

    # Run setup window if any key is missing
    if not keys_complete(env):
        setup_window.run()  # Call directly instead of subprocess!
        
        # Re-read .env — setup window saved new keys into it
        env = load_env()
        apply_env(env)

    try:
        import innostaa_pyttsx3
        innostaa_pyttsx3.main()
    except Exception as e:
        import traceback
        from tkinter import messagebox
        
        error_details = traceback.format_exc()
        messagebox.showerror("Critical Error", f"Failed to start INNOSTAA:\n\n{error_details}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()