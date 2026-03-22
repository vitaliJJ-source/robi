from dataclasses import dataclass
from typing import Optional

from robicore.capabilities import Capabilities
from robicore.providers.chain import ProviderChain


@dataclass
class Services:
    """
    Central dependency container.
    The Orchestrator receives ONE instance of this.
    """

    caps: Capabilities

    # Provider chains (can be None for now)
    stt_chain: Optional[ProviderChain] = None
    tts_chain: Optional[ProviderChain] = None
    llm_chain: Optional[ProviderChain] = None
    search_chain: Optional[ProviderChain] = None

    # You will inject these later:
    logmgr: Optional[object] = None
    memory: Optional[object] = None
    chara: Optional[object] = None
    lt_memory: Optional[object] = None
    tokens: Optional[object] = None
    audio_player: Optional[object] = None
