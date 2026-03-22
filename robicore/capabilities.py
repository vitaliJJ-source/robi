from dataclasses import dataclass


@dataclass
class Capabilities:
    """
    Runtime capability flags.
    These are dynamic and can change during runtime.
    No logic here — just state.
    """

    # Connectivity
    network: bool = False

    # Speech-to-text
    stt_online: bool = False
    stt_offline: bool = True

    # Text-to-speech
    tts_online: bool = False
    tts_offline: bool = True

    # LLM
    llm_online: bool = False
    llm_offline: bool = True

    # Tools
    web_search: bool = False
    external_channels: bool = False

    def summary(self) -> str:
        return (
            f"network={self.network} "
            f"stt_online={self.stt_online} "
            f"tts_online={self.tts_online} "
            f"llm_online={self.llm_online} "
            f"web_search={self.web_search}"
        )
