from robicore.capabilities import Capabilities
from robicore.providers.base import TTSProvider


class OpenAITtsProvider(TTSProvider):
    name = "tts_openai"
    online = True

    def __init__(self, tts_agent, voice: str):
        """
        tts_agent must implement:
          speak_async(text, voice=..., on_error=..., on_start=..., on_end=..., on_timing=...)
          stop()
        """
        self._tts = tts_agent
        self._voice = voice

    def is_available(self, caps: Capabilities) -> bool:
        return bool(caps.tts_online and caps.network)

    def speak(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        # Minimal: orchestrator can attach hooks later; for now just speak.
        self._tts.speak_async(t, voice=self._voice)
