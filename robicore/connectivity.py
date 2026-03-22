import socket
from robicore.capabilities import Capabilities


class ConnectivityMonitor:
    """
    Simple connectivity checker.
    Updates Capabilities dynamically.
    """

    def __init__(self, caps: Capabilities, test_host: str = "8.8.8.8", test_port: int = 53):
        self.caps = caps
        self.test_host = test_host
        self.test_port = test_port

    def _check_network(self, timeout: float = 1.5) -> bool:
        try:
            with socket.create_connection((self.test_host, self.test_port), timeout=timeout):
                return True
        except Exception:
            return False

    def tick(self) -> None:
        online = self._check_network()

        # Update dynamic capabilities
        self.caps.network = online

        self.caps.stt_online = online
        self.caps.tts_online = online
        self.caps.llm_online = online
        self.caps.web_search = online
        self.caps.external_channels = online
