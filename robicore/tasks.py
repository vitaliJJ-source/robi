from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    name: str
    requires_network: bool = False


class TaskEngine:
    def __init__(self, services):
        self.services = services
        self._tasks: List[Task] = []

    def tick(self) -> None:
        print(f"[TaskEngine] tick | tasks={len(self._tasks)}", flush=True)

    def add_task(self, task: Task) -> None:
        self._tasks.append(task)