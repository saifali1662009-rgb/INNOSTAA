"""
barge_in.py  —  INNOSTAA barge-in v12
===============================================================================
Fixes vs v11:
  1. _calibrate_noise no longer steals from audio_q — it uses its own
     dedicated _calib_q fed by a separate tee in audio_callback. This
     eliminates the race condition that caused session freezes.
  2. COOLDOWN_S raised to 1.2s — covers actual TTS speaker bleed into mic.
     The watcher drains _barge_q silently during cooldown so no echo frames
     ever reach the VAD scorer.
  3. Post-TTS echo wait moved here: speak() waits ECHO_FADE_S after playback
     before returning, giving the mic time to settle before listen() starts.
  4. listen() no longer needs to import barge_in mid-call; barge_audio_frames
     is a plain module-level list accessed directly.
"""

from __future__ import annotations
import asyncio, io, os, queue, threading, time, logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import edge_tts

log = logging.getLogger("barge_in")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 16_000
BLOCKSIZE      = 4_000        # must match innostaa_pyttsx3.py
VAD_CHUNK      = 512          # Silero VAD requires exactly 512 samples @ 16 kHz

# Cooldown after TTS starts — long enough to cover speaker bleed into mic.
# At 0.25 s the TTS audio was still playing and VAD scored it as speech (1.00).
COOLDOWN_S     = 1.20

NOISE_MULT     = 3.5
ENERGY_ABS_MIN = 180          # floor for quiet rooms / laptop mics
CONFIRM_FRAMES = 2
CANDIDATE_KEEP = 8            # rolling pre-confirm window (~2 s of lead-in)

# After playback ends, wait this long before returning from speak() so the
# mic doesn't immediately pick up speaker echo/reverb.
ECHO_FADE_S    = 0.40

# Noise calibration — uses its own queue, never touches audio_q
_calib_q        : queue.Queue = queue.Queue(maxsize=400)
_noise_floor    : float = 0.0
_dyn_threshold  : float = ENERGY_ABS_MIN
_calib_lock     = threading.Lock()
_calib_active   = False        # True while calibration thread is running

# Voiced frames saved during barge-in for listen() to prepend
barge_audio_frames: list = []

_VAD_CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".cache", "innostaa")
_VAD_CACHE_FILE = os.path.join(_VAD_CACHE_DIR, "silero_vad.pt")
_vad_model      = None
_vad_ready      = False

# Fed by audio_callback in innostaa_pyttsx3.py ONLY while is_speaking is set
_barge_q: queue.Queue = queue.Queue(maxsize=600)

_state: dict = {
    "is_speaking": None, "is_interrupted": None,
    "mic_enabled": lambda: True, "window": lambda: None,
    "history": None, "SRC_SYSTEM": None,
    "edge_voice": "en-US-AriaNeural", "tts_lock": threading.Lock(),
    "audio_q": None,
}

# ── install() ─────────────────────────────────────────────────────────────────

def install(*, audio_q, is_speaking, is_interrupted,
            mic_enabled_fn, window_fn, history, src_system,
            edge_voice, tts_lock,
            main_module=None, normalize_text_fn=None):

    _state.update({
        "audio_q": audio_q, "is_speaking": is_speaking,
        "is_interrupted": is_interrupted, "mic_enabled": mic_enabled_fn,
        "window": window_fn, "history": history, "SRC_SYSTEM": src_system,
        "edge_voice": edge_voice, "tts_lock": tts_lock,
    })
    _load_vad()
    # Calibrate using the dedicated calib queue (audio_callback tees into it)
    threading.Thread(target=_calibrate_noise, daemon=True).start()
    print("[barge_in] v12 installed.", flush=True)


# ── Background noise calibration (uses _calib_q, never audio_q) ──────────────

def _calibrate_noise(duration=1.5):
    """Read from _calib_q for `duration` seconds, set noise floor."""
    global _noise_floor, _dyn_threshold, _calib_active
    _calib_active = True
    vals = []
    deadline = time.time() + duration
    while time.time() < deadline:
        try:
            blk = _calib_q.get(timeout=0.1)
            vals.append(float(np.abs(np.frombuffer(blk, np.int16)).mean()))
        except Exception:
            continue
    _calib_active = False
    if vals:
        floor = float(np.median(vals))
        with _calib_lock:
            _noise_floor   = floor
            _dyn_threshold = max(floor * NOISE_MULT, ENERGY_ABS_MIN)
        print(f"[barge_in] calibrated: noise_floor={floor:.0f}  "
              f"dyn_threshold={_dyn_threshold:.0f}  samples={len(vals)}", flush=True)


# ── VAD ───────────────────────────────────────────────────────────────────────

def _load_vad():
    global _vad_model, _vad_ready
    if _vad_ready:
        return
    try:
        import torch
        os.makedirs(_VAD_CACHE_DIR, exist_ok=True)
        if os.path.exists(_VAD_CACHE_FILE):
            _vad_model = torch.jit.load(_VAD_CACHE_FILE)
            _vad_model.eval()
            _vad_ready = True
            print("[barge_in] Silero VAD loaded from cache.", flush=True)
            return
        print("[barge_in] Downloading Silero VAD (~2 MB, once only)...", flush=True)
        import urllib.request, zipfile
        zip_path = os.path.join(_VAD_CACHE_DIR, "_tmp.zip")
        urllib.request.urlretrieve(
            "https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip",
            zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            name = next(n for n in zf.namelist() if n.endswith("silero_vad.jit"))
            with zf.open(name) as src, open(_VAD_CACHE_FILE, "wb") as dst:
                dst.write(src.read())
        os.remove(zip_path)
        _vad_model = torch.jit.load(_VAD_CACHE_FILE)
        _vad_model.eval()
        _vad_ready = True
        print("[barge_in] Silero VAD saved.", flush=True)
    except Exception as e:
        print(f"[barge_in] Silero unavailable ({e}); amplitude fallback.", flush=True)


def _vad_score(pcm: np.ndarray) -> float:
    """Score a block. Splits into 512-sample chunks; returns max score."""
    if _vad_ready and _vad_model is not None:
        try:
            import torch
            scores = []
            for i in range(0, len(pcm) - VAD_CHUNK + 1, VAD_CHUNK):
                chunk = pcm[i:i + VAD_CHUNK].astype(np.float32) / 32768.0
                wav   = torch.from_numpy(chunk).unsqueeze(0)
                with torch.no_grad():
                    out = _vad_model(wav, SAMPLE_RATE)
                    scores.append(float(out.item() if out.dim() == 0 else out[0].item()))
            if scores:
                return max(scores)
        except Exception:
            pass
    return 1.0 if float(np.abs(pcm).mean()) >= _dyn_threshold else 0.0


# ── speak() ───────────────────────────────────────────────────────────────────

def speak(text: str):
    global barge_audio_frames

    aq      = _state["audio_q"]
    is_spk  = _state["is_speaking"]
    is_int  = _state["is_interrupted"]
    mic_ok  = _state["mic_enabled"]
    get_win = _state["window"]
    hist    = _state["history"]
    src_sys = _state["SRC_SYSTEM"]
    voice   = _state["edge_voice"]
    lock    = _state["tts_lock"]

    print("Assistant:", text)
    if hist and src_sys is not None:
        hist.add("assistant", text, source=src_sys)

    safe = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ").replace("\r", "")
    win = get_win()
    if win:
        try:
            win.evaluate_js(f"updateStatus('speaking','{safe}')")
            win.evaluate_js(f"addMessage('assistant','{safe}')")
        except Exception:
            pass

    is_int.clear()
    barge_audio_frames = []
    is_spk.set()
    _drain(aq)
    _drain(_barge_q)

    # ── Events / shared result ────────────────────────────────────────────────
    _synth_done   = threading.Event()
    _play_done    = threading.Event()
    _play_started = threading.Event()   # set the instant sd.play() is called
    _result       = {}

    # ── Synthesis (background so watcher starts immediately) ─────────────────
    def _synth_thread():
        try:
            async def _synth():
                buf = b""
                async for c in edge_tts.Communicate(text, voice=voice).stream():
                    if c["type"] == "audio":
                        buf += c["data"]
                return buf
            loop = asyncio.new_event_loop()
            mp3  = loop.run_until_complete(_synth())
            loop.close()
            if not mp3:
                _result["error"] = "empty"
                return
            pcm_f32, sr = sf.read(io.BytesIO(mp3), dtype="float32")
            if pcm_f32.ndim > 1:
                pcm_f32 = pcm_f32.mean(axis=1)
            _result["pcm"] = pcm_f32
            _result["sr"]  = sr
        except Exception as e:
            _result["error"] = str(e)
            print(f"[barge_in] synth error: {e}", flush=True)
        finally:
            _synth_done.set()

    # ── Watcher ───────────────────────────────────────────────────────────────
    def _watcher():
        # Wait until audio is actually coming out of the speaker before
        # starting the cooldown clock. Without this, the cooldown expires
        # during synthesis (network latency) before any TTS audio plays.
        _play_started.wait()
        cooldown_end   = time.time() + COOLDOWN_S
        frames_above   = 0
        candidate_blks: list = []

        while not _play_done.is_set():
            try:
                blk = _barge_q.get(timeout=0.05)
            except queue.Empty:
                continue

            # During cooldown: drain silently — never score these frames.
            # This is what prevents TTS speaker bleed from triggering barge-in.
            if time.time() < cooldown_end:
                continue

            pcm    = np.frombuffer(blk, np.int16)
            energy = float(np.abs(pcm).mean())
            score  = _vad_score(pcm)

            # Dual gate: Silero confidence AND amplitude above calibrated floor
            voiced = (score >= 0.40) and (energy >= _dyn_threshold * 0.6)

            if voiced:
                candidate_blks.append(blk)
                if len(candidate_blks) > CANDIDATE_KEEP:
                    candidate_blks.pop(0)
                frames_above += 1
                if frames_above >= CONFIRM_FRAMES:
                    global barge_audio_frames
                    barge_audio_frames = list(candidate_blks)
                    is_int.set()
                    is_spk.clear()
                    sd.stop()
                    print(f"[barge_in] Interrupted! "
                          f"energy={energy:.0f}  vad={score:.2f}  "
                          f"saved={len(barge_audio_frames)}", flush=True)
                    _play_done.set()
                    return
            else:
                candidate_blks = candidate_blks[-(CANDIDATE_KEEP // 2):]
                frames_above   = max(0, frames_above - 1)

    # ── Playback ──────────────────────────────────────────────────────────────
    def _play():
        _synth_done.wait()
        if is_int.is_set() or "error" in _result or "pcm" not in _result:
            _play_started.set()   # unblock watcher even on error
            _play_done.set()
            return
        try:
            with lock:
                if not is_int.is_set():
                    _play_started.set()   # cooldown clock starts NOW
                    sd.play(_result["pcm"], _result["sr"])
                    sd.wait()
                else:
                    _play_started.set()
        except Exception as e:
            log.debug(f"play: {e}")
        finally:
            _play_done.set()

    synth_t = threading.Thread(target=_synth_thread, daemon=True)
    watch_t = threading.Thread(target=_watcher,      daemon=True)
    play_t  = threading.Thread(target=_play,         daemon=True)

    synth_t.start()
    watch_t.start()
    play_t.start()

    play_t.join()
    is_spk.clear()
    _play_done.set()
    watch_t.join(timeout=0.5)

    if not is_int.is_set():
        # Wait for speaker echo to fade before returning to listen()
        time.sleep(ECHO_FADE_S)
        _drain(aq)
        _drain(_barge_q)
        # Recalibrate using dedicated calib queue — does NOT touch audio_q
        threading.Thread(target=_calibrate_noise, kwargs={"duration": 0.8},
                         daemon=True).start()

    if win:
        try:
            status = "listening" if mic_ok() else "idle"
            win.evaluate_js(f"updateStatus('{status}','')")
        except Exception:
            pass


def _drain(q: queue.Queue):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break