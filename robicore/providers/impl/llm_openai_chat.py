from typing import Any, Dict, List, Optional

from robicore.capabilities import Capabilities
from robicore.providers.base import LLMProvider


class OpenAIChatLlmProvider(LLMProvider):
    name = "llm_openai_chat"
    online = True

    def __init__(self, openai_client, model: str, temperature: float = 0.2):
        self._client = openai_client
        self._model = model
        self._temp = float(temperature)

    def is_available(self, caps: Capabilities) -> bool:
        return bool(caps.llm_online and caps.network)

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        # kwargs may include: max_tokens, temperature override, etc.
        temperature = float(kwargs.get("temperature", self._temp))
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text
