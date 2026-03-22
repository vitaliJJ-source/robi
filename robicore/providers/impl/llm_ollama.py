from typing import Any, Dict, List

from robicore.capabilities import Capabilities
from robicore.providers.base import LLMProvider


def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    # Minimal prompt assembly: keep last turns short and stable
    parts: List[str] = []
    for m in messages[-12:]:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}")
        else:
            parts.append(f"[USER]\n{content}")
    return "\n\n".join(parts).strip()


class OllamaLlmProvider(LLMProvider):
    name = "llm_ollama"
    online = False

    def __init__(self, ollama_agent, system: str = "", timeout_sec: float = 50.0):
        """
        ollama_agent must implement:
          chat(user_text: str, system: str, timeout: float) -> str
        """
        self._agent = ollama_agent
        self._system = (system or "").strip()
        self._timeout = float(timeout_sec)

    def is_available(self, caps: Capabilities) -> bool:
        return bool(caps.llm_offline)

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        prompt = _messages_to_prompt(messages)
        if not prompt:
            return ""
        out = self._agent.chat(prompt, system=self._system, timeout=self._timeout)
        return (out or "").strip()
