import subprocess
import sys
from typing import Iterable

MODULES = {
    "e3m": "servers.e3m_reward_server",
    "world_model": "servers.world_model_reward_server",
    "external": "servers.external_reward_server",
    "actions": "servers.action_executor_server",
}


class Orchestrator:
    """Launch and stop server modules based on a list of names."""

    def __init__(self, modules: Iterable[str]):
        self.module_names = list(modules)
        self.processes: list[subprocess.Popen] = []

    def start(self) -> None:
        for name in self.module_names:
            mod = MODULES.get(name)
            if not mod:
                continue
            cmd = [sys.executable, '-m', mod]
            p = subprocess.Popen(cmd)
            self.processes.append(p)

    def stop(self) -> None:
        for p in self.processes:
            p.terminate()
            try:
                p.wait(timeout=1)
            except Exception:
                pass
        self.processes.clear()
