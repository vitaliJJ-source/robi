from typing import Optional, Tuple

from robicore.capabilities import Capabilities
from robicore.providers.base import STTProvider


class OpenAISttProvider(STTProvider):
    name = "stt_openai"
    online = True

    def __init__(self, stt_agent, language_hint: Optional[str] = None):
        """
        stt_agent must implement:
          transcribe_wav_bytes(wav_bytes: bytes, language_hint: Optional[str]) -> (text, ms, toks)
        """
        self._agent = stt_agent
        self._lang = (language_hint or "").strip() or None

    def is_available(self, caps: Capabilities) -> bool:
        return bool(caps.stt_online and caps.network)

    def transcribe(self, wav_bytes: bytes) -> Tuple[str, float, int]:
        text, ms, toks = self._agent.transcribe_wav_bytes(wav_bytes, language_hint=self._lang)
        return (text or "").strip(), float(ms or 0.0), int(toks or 0)
