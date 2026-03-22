from robicore.capabilities import Capabilities
from robicore.connectivity import ConnectivityMonitor
from robicore.providers.chain import ProviderChain
from robicore.services import Services


def build_services_skeleton() -> tuple[Services, ConnectivityMonitor]:
    """
    Step-1 factory: creates Services + ConnectivityMonitor with offline-first defaults.
    Provider chains are left None until you inject your real agents in Step-2.
    """
    caps = Capabilities()  # offline-first defaults
    services = Services(caps=caps)
    monitor = ConnectivityMonitor(caps=caps)
    return services, monitor


def attach_chains(
    services: Services,
    *,
    stt_providers=None,
    tts_providers=None,
    llm_providers=None,
    search_providers=None,
) -> Services:
    """
    Attach provider chains in priority order.
    """
    if stt_providers is not None:
        services.stt_chain = ProviderChain(list(stt_providers))
    if tts_providers is not None:
        services.tts_chain = ProviderChain(list(tts_providers))
    if llm_providers is not None:
        services.llm_chain = ProviderChain(list(llm_providers))
    if search_providers is not None:
        services.search_chain = ProviderChain(list(search_providers))
    return services
