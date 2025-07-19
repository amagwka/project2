from __future__ import annotations
import subprocess
from typing import Iterable, Optional
from urllib.parse import urlparse

from .orchestrator import Orchestrator


DEFAULT_MODULES = (
    "actions",
    "external",
    "world_model",
    "e3m",
)


class ServerManager:
    """Start and stop the default set of NATS helper servers."""

    def __init__(self, modules: Optional[Iterable[str]] = None):
        self._modules = list(modules) if modules is not None else list(DEFAULT_MODULES)
        self._orchestrator = Orchestrator(self._modules)
        self._nats_proc: Optional[subprocess.Popen] = None

    def _get_port(self, url: str) -> int:
        parsed = urlparse(url)
        if parsed.port is not None:
            return parsed.port
        # Fallback to default NATS port
        try:
            return int(parsed.path.split(":")[-1])
        except Exception:
            return 4222

    def start(self, env: Optional[object] = None) -> None:
        """Launch ``nats-server`` and the configured modules."""
        port = 4222
        if env is not None and hasattr(env, "nats_url"):
            try:
                port = self._get_port(env.nats_url)
            except Exception:
                port = 4222

        if self._nats_proc is None:
            cmd = ["nats-server", "-p", str(port)]
            self._nats_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self._orchestrator.start()

    def stop(self) -> None:
        """Terminate all started processes."""
        self._orchestrator.stop()
        if self._nats_proc is not None:
            self._nats_proc.terminate()
            try:
                self._nats_proc.wait(timeout=5)
            except Exception:
                pass
            self._nats_proc = None
