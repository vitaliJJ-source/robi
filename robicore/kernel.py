# robicore/kernel.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class KernelTimers:
    net_tick_sec: float = 20.0
    fast_tick_sec: float = 0.25
    orchestrator_tick_sec: float = 15 * 60.0  # 15 min


class Kernel:
    """
    Minimal kernel loop for orchestration:
    - ticks connectivity
    - runs a 15-min orchestration hook
    - later: task scheduler, proactive prompts, IO tools, etc.
    """
    def __init__(self, services, netmon, timers: Optional[KernelTimers] = None, on_net_change: Optional[Callable[[bool], None]] = None):
        self.services = services
        self.netmon = netmon
        self.timers = timers or KernelTimers()
        self.on_net_change = on_net_change

        now = time.monotonic()
        self._next_net = now
        self._next_orch = now + self.timers.orchestrator_tick_sec
        self._last_net: Optional[bool] = None

    def tick(self) -> None:
        now = time.monotonic()

        # Connectivity tick
        if now >= self._next_net:
            self._next_net = now + max(2.0, float(self.timers.net_tick_sec))
            try:
                self.netmon.tick()
            except Exception:
                pass

            cur = bool(getattr(self.services.caps, "network", False))
            if self._last_net is None or cur != self._last_net:
                self._last_net = cur
                if self.on_net_change:
                    try:
                        self.on_net_change(cur)
                    except Exception:
                        pass

        # 15-minute orchestrator tick (placeholder)
        if now >= self._next_orch:
            self._next_orch = now + max(30.0, float(self.timers.orchestrator_tick_sec))
            try:
                self.tick_15min()
            except Exception:
                pass

    def tick_15min(self) -> None:
        # Placeholder for future:
        # - task queue check
        # - proactive outreach conditions
        # - email/whatsapp dispatch
        # - device housekeeping
        return
