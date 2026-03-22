from robicore.capabilities import Capabilities
from robicore.providers.base import SearchProvider


class OpenAISearchProvider(SearchProvider):
    name = "search_openai"
    online = True

    def __init__(self, search_agent):
        """
        search_agent must implement:
          search(query: str) -> (text, ms, toks)   OR just text
        """
        self._agent = search_agent

    def is_available(self, caps: Capabilities) -> bool:
        return bool(caps.web_search and caps.network)

    def search(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return ""
        out = self._agent.search(q)
        if isinstance(out, tuple) and len(out) >= 1:
            return (out[0] or "").strip()
        return (out or "").strip()
