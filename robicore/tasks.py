from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Task:
    name: str
    requires_network: bool = False

    def tick(self, services) -> None:
        pass


class HeartbeatTask(Task):
    def __init__(self):
        super().__init__(name="heartbeat", requires_network=False)

    def tick(self, services) -> None:
        print(f"[{_ts()}] [Task:heartbeat] alive", flush=True)


class NetworkHeartbeatTask(Task):
    def __init__(self):
        super().__init__(name="network_heartbeat", requires_network=True)

    def tick(self, services) -> None:
        print(f"[{_ts()}] [Task:network_heartbeat] online task ran", flush=True)


class TaskEngine:
    def __init__(self, services):
        self.services = services
        self._tasks: List[Task] = [
            HeartbeatTask(),
            NetworkHeartbeatTask(),
        ]

    def tick(self, services: Optional[object] = None) -> None:
        active_services = services or self.services
        print(f"[{_ts()}] [TaskEngine] tick | tasks={len(self._tasks)}", flush=True)

        for task in self._tasks:
            try:
                if task.requires_network and not bool(getattr(active_services.caps, "network", False)):
                    print(
                        f"[{_ts()}] [TaskEngine] skipping task: {task.name} (offline)",
                        flush=True,
                    )
                    continue

                print(f"[{_ts()}] [TaskEngine] running task: {task.name}", flush=True)
                task.tick(active_services)
            except Exception as e:
                print(f"[{_ts()}] [TaskEngine] task '{task.name}' error: {e}", flush=True)

    def add_task(self, task: Task) -> None:
        self._tasks.append(task)