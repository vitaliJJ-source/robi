from typing import Tuple

from robicore.capabilities import Capabilities
from robicore.providers.base import STTProvider


class VoskSttProvider(STTProvider):
    name = "stt_vosk"
    online = False

    def __init__(self, vosk_agent):
        """
        vosk_agent must implement:
          transcribe_wav_bytes(wav_bytes: bytes) -> (text, ms, toks)
        """
        self._agent = vosk_agent

    def is_available(self, caps: Capabilities) -> bool:
        return bool(caps.stt_offline)

    def transcribe(self, wav_bytes: bytes) -> Tuple[str, float, int]:
        text, ms, toks = self._agent.transcribe_wav_bytes(wav_bytes)
        return (text or "").strip(), float(ms or 0.0), int(toks or 0)
