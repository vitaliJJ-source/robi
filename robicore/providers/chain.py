from typing import List, Optional, TypeVar, Generic

from robicore.capabilities import Capabilities

T = TypeVar("T")


class ProviderChain(Generic[T]):
    """
    Ordered provider selection.
    First available provider wins.
    """

    def __init__(self, providers: List[T]):
        self.providers = providers

    def get(self, caps: Capabilities) -> Optional[T]:
        for provider in self.providers:
            try:
                if provider.is_available(caps):
                    return provider
            except Exception:
                continue
        return None

    def all_available(self, caps: Capabilities) -> List[T]:
        out = []
        for provider in self.providers:
            try:
                if provider.is_available(caps):
                    out.append(provider)
            except Exception:
                continue
        return out
