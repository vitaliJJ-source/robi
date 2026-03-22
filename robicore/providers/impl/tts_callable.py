from typing import Callable

from robicore.capabilities import Capabilities
from robicore.providers.base import TTSProvider


class CallableTtsProvider(TTSProvider):
    """
    Wrap any function speak_fn(text:str)->None as a TTS provider.
    Perfect for your current _speak_text_local implementation.
    """
    name = "tts_callable"
    online = False

    def __init__(self, speak_fn: Callable[[str], None], name: str = "tts_callable"):
        self.name = name
        self._speak_fn = speak_fn

    def is_available(self, caps: Capabilities) -> bool:
        return bool(caps.tts_offline)

    def speak(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        self._speak_fn(t)
