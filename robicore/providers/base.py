from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

from robicore.capabilities import Capabilities


# -----------------------
# STT
# -----------------------

class STTProvider(ABC):
    name: str
    online: bool  # True = requires network

    @abstractmethod
    def is_available(self, caps: Capabilities) -> bool:
        ...

    @abstractmethod
    def transcribe(self, wav_bytes: bytes) -> Tuple[str, float, int]:
        """
        Returns: (text, latency_ms, tokens)
        """
        ...


# -----------------------
# TTS
# -----------------------

class TTSProvider(ABC):
    name: str
    online: bool

    @abstractmethod
    def is_available(self, caps: Capabilities) -> bool:
        ...

    @abstractmethod
    def speak(self, text: str) -> None:
        ...


# -----------------------
# LLM
# -----------------------

class LLMProvider(ABC):
    name: str
    online: bool

    @abstractmethod
    def is_available(self, caps: Capabilities) -> bool:
        ...

    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        ...


# -----------------------
# Web Search
# -----------------------

class SearchProvider(ABC):
    name: str
    online: bool = True  # always online in practice

    @abstractmethod
    def is_available(self, caps: Capabilities) -> bool:
        ...

    @abstractmethod
    def search(self, query: str) -> str:
        ...
