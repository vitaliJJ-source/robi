#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import sys
import time
import json
import threading
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

# --- robicore kernel pieces ---
from robicore.factory import build_services_skeleton, attach_chains
from robicore.providers.impl.stt_openai import OpenAISttProvider
from robicore.providers.impl.stt_vosk import VoskSttProvider
from robicore.providers.impl.tts_openai import OpenAITtsProvider
from robicore.providers.impl.tts_callable import CallableTtsProvider
from robicore.providers.impl.llm_openai_chat import OpenAIChatLlmProvider
from robicore.providers.impl.llm_ollama import OllamaLlmProvider
from robicore.kernel import Kernel, KernelTimers
from robicore.tasks import TaskEngine

# Optional: Auto-VAD
try:
    import webrtcvad  # type: ignore
    HAVE_WEBRTCVAD = True
except Exception:
    webrtcvad = None
    HAVE_WEBRTCVAD = False

# Optional: Vosk STT
try:
    import vosk  # type: ignore
    HAVE_VOSK = True
except Exception:
    vosk = None
    HAVE_VOSK = False

# OpenAI
from openai import OpenAI


# =========================
# Config (minimal)
# =========================

def _env_true(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in ("1", "true", "True", "yes", "YES")

# Audio
AUDIO_SR = int(os.getenv("AUDIO_SR", "16000"))
PLAYBACK_SR = int(os.getenv("PLAYBACK_SR", "48000"))
AUDIO_IN_DEVICE = os.getenv("AUDIO_IN_DEVICE")   # optional name substring or index string
AUDIO_OUT_DEVICE = os.getenv("AUDIO_OUT_DEVICE") # optional

# VAD
AUTO_VAD = os.getenv("AUTO_VAD", "1") not in ("0", "false", "False", "no", "NO")
VAD_MODE = int(os.getenv("VAD_MODE", "2"))            # 0-3
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", "30"))   # 10/20/30
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "900"))
VAD_PRE_MS = int(os.getenv("VAD_PRE_MS", "300"))
VAD_START_FRAMES = int(os.getenv("VAD_START_FRAMES", "4"))
VAD_MIN_RMS = float(os.getenv("VAD_MIN_RMS", "0.006"))

# STT models
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
DEFAULT_LANG_CODE = (os.getenv("DEFAULT_LANG_CODE", "en") or "en").strip().lower()

OFFLINE_STT = _env_true("OFFLINE_STT", "1")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "/home/vjudin/vosk-models/vosk-model-small-en-us-0.15")

# LLM models
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OFFLINE_LLM = os.getenv("OLLAMA_MODEL", "gemma3:270m")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "50"))

# TTS
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.2"))
OPENAI_TTS_FORMAT = os.getenv("OPENAI_TTS_FORMAT", "wav")

OFFLINE_TTS = _env_true("OFFLINE_TTS", "1")
OFFLINE_TTS_BIN = os.getenv("OFFLINE_TTS_BIN", "espeak-ng")
OFFLINE_TTS_VOICE = os.getenv("OFFLINE_TTS_VOICE", "en-us+f3")
OFFLINE_TTS_RATE = int(os.getenv("OFFLINE_TTS_RATE", "150"))
OFFLINE_TTS_PITCH = int(os.getenv("OFFLINE_TTS_PITCH", "60"))
OFFLINE_TTS_VOLUME = int(os.getenv("OFFLINE_TTS_VOLUME", "170"))

# Connectivity tick
NET_TICK_SEC = float(os.getenv("NET_TICK_SEC", "20"))


# =========================
# Small helpers
# =========================

def _t_ms() -> float:
    return time.perf_counter() * 1000.0

def pick_device_id(kind: str, want: Optional[str]) -> Optional[int]:
    if not want:
        return None
    want = want.strip()
    if want.isdigit():
        return int(want)
    devs = sd.query_devices()
    wlow = want.lower()
    for i, d in enumerate(devs):
        if kind == "output" and d.get("max_output_channels", 0) <= 0:
            continue
        if kind == "input" and d.get("max_input_channels", 0) <= 0:
            continue
        if wlow in (d.get("name", "") or "").lower():
            return i
    return None

def resample_linear_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.ascontiguousarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n_in = len(x)
    if n_in <= 1:
        return np.ascontiguousarray(x, dtype=np.float32)
    n_out = max(1, int(round(n_in * (float(sr_out) / float(sr_in)))))
    xp = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    xq = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
    y = np.interp(xq, xp, x).astype(np.float32)
    return np.ascontiguousarray(y, dtype=np.float32)


# =========================
# Minimal STT agents
# =========================

class STTAgentOpenAI:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def transcribe_wav_bytes(self, wav_bytes: bytes, language_hint: Optional[str] = None) -> Tuple[str, float, int]:
        t0 = _t_ms()
        f = io.BytesIO(wav_bytes)
        f.name = "utterance.wav"
        kwargs: Dict[str, Any] = {"model": self.model, "file": f}
        if language_hint:
            kwargs["language"] = language_hint
        resp = self.client.audio.transcriptions.create(**kwargs)
        text = (resp.text or "").strip()
        return text, (_t_ms() - t0), 0


class STTAgentVosk:
    def __init__(self, model_path: str):
        if not HAVE_VOSK:
            raise RuntimeError("vosk is not installed")
        if not os.path.isdir(model_path):
            raise RuntimeError(f"Vosk model not found: {model_path}")
        self.model = vosk.Model(model_path)

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> Tuple[str, float, int]:
        t0 = _t_ms()
        buf = io.BytesIO(wav_bytes)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1).astype(np.float32)
        if sr != 16000:
            audio = resample_linear_mono(audio, sr, 16000)
            sr = 16000
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        rec = vosk.KaldiRecognizer(self.model, 16000)
        rec.SetWords(False)
        rec.AcceptWaveform(pcm16)
        j = json.loads(rec.FinalResult() or "{}")
        text = (j.get("text") or "").strip()
        return text, (_t_ms() - t0), 0


# =========================
# Minimal TTS agents
# =========================

class TTSAgentOpenAI:
    def __init__(self, client: OpenAI, model: str, voice: str, speed: float, audio_format: str):
        self.client = client
        self.model = model
        self.voice = voice
        self.speed = float(speed)
        self.audio_format = audio_format

    def speak_async(self, text: str, voice: Optional[str] = None) -> None:
        # very simple: generate full audio and play synchronously in a thread
        chosen_voice = voice or self.voice
        t = (text or "").strip()
        if not t:
            return

        def worker():
            try:
                resp = self.client.audio.speech.create(
                    model=self.model,
                    voice=chosen_voice,
                    input=t,
                    response_format=self.audio_format,
                    speed=self.speed,
                )
                audio_bytes = resp.read() if hasattr(resp, "read") else getattr(resp, "content", b"")
                buf = io.BytesIO(audio_bytes)
                data, sr = sf.read(buf, dtype="float32")
                if data.ndim == 2:
                    data = np.mean(data, axis=1).astype(np.float32)
                if sr != PLAYBACK_SR:
                    data = resample_linear_mono(data, sr, PLAYBACK_SR)
                    sd.play(data, samplerate=PLAYBACK_SR, blocking=True, latency="high")
            except Exception as e:
                print(f"[TTS OpenAI] error: {e}", file=sys.stderr)

        threading.Thread(target=worker, daemon=True).start()


def speak_espeak_blocking(text: str) -> None:
    t = (text or "").strip()
    if not t:
        return
    tmp = f"/tmp/robi_offline_tts_{os.getpid()}_{int(time.time()*1000)}.wav"
    cmd = [
        OFFLINE_TTS_BIN,
        "-v", str(OFFLINE_TTS_VOICE),
        "-s", str(int(OFFLINE_TTS_RATE)),
        "-p", str(int(OFFLINE_TTS_PITCH)),
        "-a", str(int(OFFLINE_TTS_VOLUME)),
        "-w", tmp,
        t,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data, sr = sf.read(tmp, dtype="float32")
    try:
        os.unlink(tmp)
    except Exception:
        pass
    if data.ndim == 2:
        data = np.mean(data, axis=1).astype(np.float32)
    if sr != PLAYBACK_SR:
        data = resample_linear_mono(data, sr, PLAYBACK_SR)
    sd.play(data, samplerate=PLAYBACK_SR, blocking=True)


# =========================
# Minimal Ollama agent
# =========================

class OllamaAgent:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, user_text: str, system: str = "", timeout: float = 30.0) -> str:
        url = self.base_url + "/api/chat"
        payload = {
            "model": self.model,
            "keep_alive": "10m",
            "stream": False,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": user_text}]
            ),
            "options": {},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8", errors="ignore"))
        return ((out.get("message") or {}).get("content") or "").strip()


# =========================
# UnifiedMic minimal (auto VAD if available)
# =========================

class UnifiedMic:
    def __init__(self, sr: int, frame_ms: int, on_utterance):
        self.sr = int(sr)
        self.frame_ms = int(frame_ms)
        self.frame_samples = int(self.sr * self.frame_ms / 1000)
        self.on_utterance = on_utterance

        self.have_vad = HAVE_WEBRTCVAD and AUTO_VAD
        self.vad = webrtcvad.Vad(VAD_MODE) if self.have_vad else None

        self.pre_frames = max(1, int(VAD_PRE_MS / self.frame_ms))
        self.silence_frames = max(1, int(VAD_SILENCE_MS / self.frame_ms))
        self.start_frames_req = max(1, int(VAD_START_FRAMES))

        self._ring: List[bytes] = []
        self._buf: List[bytes] = []
        self._in_speech = False
        self._sil = 0
        self._start_count = 0

        in_dev = pick_device_id("input", AUDIO_IN_DEVICE)

        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=in_dev,
            callback=self._cb,
        )
        self.stream.start()

    def _frame_rms(self, mono_i16: np.ndarray) -> float:
        x = mono_i16.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x)) + 1e-12)

    def _pcm16_to_wav(self, pcm_bytes: bytes) -> bytes:
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0).copy()
        buf = io.BytesIO()
        sf.write(buf, audio_f32, self.sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _emit(self, pcm_bytes: bytes) -> None:
        wav = self._pcm16_to_wav(pcm_bytes)
        self.on_utterance(wav)

    def _cb(self, indata, frames, time_info, status):
        mono = indata.reshape(-1)
        pcm = mono.tobytes()

        if not self.have_vad or self.vad is None:
            # No VAD: push fixed chunks (simple push-to-talk would go here later)
            return

        # cheap noise gate before speech start
        if self._frame_rms(mono) < VAD_MIN_RMS and not self._in_speech:
            self._ring.clear()
            self._start_count = 0
            return

        self._ring.append(pcm)
        if len(self._ring) > self.pre_frames:
            self._ring.pop(0)

        try:
            is_speech = self.vad.is_speech(pcm, self.sr)
        except Exception:
            is_speech = False

        if not self._in_speech:
            self._start_count = (self._start_count + 1) if is_speech else 0
            if self._start_count >= self.start_frames_req:
                self._in_speech = True
                self._buf = list(self._ring)
                self._sil = 0
                self._start_count = 0
        else:
            self._buf.append(pcm)
            if is_speech:
                self._sil = 0
            else:
                self._sil += 1

            if self._sil >= self.silence_frames:
                pcm_all = b"".join(self._buf)
                self._buf.clear()
                self._ring.clear()
                self._in_speech = False
                self._sil = 0
                self._emit(pcm_all)

    def close(self):
        try:
            self.stream.stop()
        finally:
            self.stream.close()


# =========================
# Minimal Orchestrator App
# =========================

class RobiApp:
    def __init__(self):
        # Configure output device early (used by sd.play)
        out_dev = pick_device_id("output", AUDIO_OUT_DEVICE)
        if out_dev is not None:
            sd.default.device = (sd.default.device[0], out_dev)

        # Build services skeleton + connectivity monitor (offline-first caps)
        self.services, self.netmon = build_services_skeleton()
        self.task_engine = TaskEngine(self.services)

        # Create providers / agents
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")) if os.environ.get("OPENAI_API_KEY") else None
        self.ollama = OllamaAgent(OLLAMA_URL, OFFLINE_LLM)

        # STT agents
        self.stt_openai = STTAgentOpenAI(self.client, OPENAI_STT_MODEL) if self.client else None
        self.stt_vosk = None
        if OFFLINE_STT and HAVE_VOSK:
            try:
                self.stt_vosk = STTAgentVosk(VOSK_MODEL_PATH)
            except Exception as e:
                print(f"[Vosk] disabled: {e}", file=sys.stderr)

        # TTS agents
        self.tts_openai = TTSAgentOpenAI(self.client, OPENAI_TTS_MODEL, OPENAI_TTS_VOICE, OPENAI_TTS_SPEED, OPENAI_TTS_FORMAT) if self.client else None

        # Build chains (priority order)
        stt_providers = []
        if self.stt_openai:
            stt_providers.append(OpenAISttProvider(self.stt_openai, language_hint=DEFAULT_LANG_CODE))
        if self.stt_vosk:
            stt_providers.append(VoskSttProvider(self.stt_vosk))

        tts_providers = []
        if OFFLINE_TTS:
            tts_providers.append(CallableTtsProvider(speak_espeak_blocking, name="tts_espeak"))
        if self.tts_openai:
            tts_providers.append(OpenAITtsProvider(self.tts_openai, voice=OPENAI_TTS_VOICE))

        llm_providers = []
        if self.client:
            llm_providers.append(OpenAIChatLlmProvider(self.client, model=OPENAI_CHAT_MODEL, temperature=0.2))
        llm_providers.append(OllamaLlmProvider(self.ollama, system="Offline companion mode. Be brief and warm.", timeout_sec=OLLAMA_TIMEOUT_SEC))

        attach_chains(
            self.services,
            stt_providers=stt_providers,
            tts_providers=tts_providers,
            llm_providers=llm_providers,
        )

        # NEW: Kernel (replaces _net_loop thread)
        self.kernel = Kernel(
            services=self.services,
            netmon=self.netmon,
            timers=KernelTimers(
                net_tick_sec=NET_TICK_SEC,
                fast_tick_sec=0.25,
                orchestrator_tick_sec=5 * 60
            ),
            on_net_change=lambda online: print(
                f"[Robi] network={online} | caps: {self.services.caps.summary()}",
                flush=True
            ),
            task_engine=self.task_engine,
        )


        # Start mic
        self.mic = UnifiedMic(sr=AUDIO_SR, frame_ms=VAD_FRAME_MS, on_utterance=self.on_utterance)


        print("[Robi] minimal core started (offline-first). Speak into the mic.", flush=True)


    def on_utterance(self, wav_bytes: bytes):
        threading.Thread(target=self._handle_utterance, args=(wav_bytes,), daemon=True).start()

    def _handle_utterance(self, wav_bytes: bytes):
        # 1) STT
        stt = self.services.stt_chain.get(self.services.caps) if self.services.stt_chain else None
        if stt is None:
            print("[Robi] No STT provider available.", flush=True)
            return

        print(f"[Robi] STT provider: {getattr(stt, 'name', type(stt).__name__)}", flush=True)

        try:
            text, ms, _ = stt.transcribe(wav_bytes)
        except Exception as e:
            print(f"[Robi] STT error: {e}", flush=True)
            return

        text = (text or "").strip()
        if not text:
            return

        print(f"[You] {text}", flush=True)

        # 2) LLM
        llm = self.services.llm_chain.get(self.services.caps) if self.services.llm_chain else None
        if llm is None:
            print("[Robi] No LLM provider available.", flush=True)
            return

        print(f"[Robi] LLM provider: {getattr(llm, 'name', type(llm).__name__)}", flush=True)

        messages = [
            {"role": "system", "content": "You are Robi, a friendly companion on a Raspberry Pi. Keep responses short and voice-friendly."},
            {"role": "user", "content": text},
        ]

        try:
            reply = llm.generate(messages)
        except Exception as e:
            print(f"[Robi] LLM error: {e}", flush=True)
            return

        reply = (reply or "").strip()
        if not reply:
            reply = "Hmm. I’m here."

        print(f"[Robi] {reply}", flush=True)

        # 3) TTS
        tts = self.services.tts_chain.get(self.services.caps) if self.services.tts_chain else None
        if tts is None:
            print("[Robi] No TTS provider available.", flush=True)
            return
        
        print(
            f"[Robi] TTS provider: {getattr(tts, 'name', type(tts).__name__)}",
            flush=True
        )
        
        try:
            tts.speak(reply)
            print("[Robi] TTS speak() called", flush=True)
        except Exception as e:
            print(f"[Robi] TTS error: {e}", flush=True)
    
    
    def close(self):
        try:
            self.mic.close()
        except Exception:
            pass


def main():
    try:
        app = RobiApp()
        while True:
            app.kernel.tick()
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[Robi] shutting down...", flush=True)
    finally:
        try:
            app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()