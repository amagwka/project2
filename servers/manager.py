import socket
import subprocess
import sys
from typing import List


class ServerManager:
    """Launch and stop reward/action/world model servers."""

    def __init__(self) -> None:
        self._processes: List[subprocess.Popen] = []

    def _port_in_use(self, addr) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(addr)
        except OSError:
            sock.close()
            return True
        sock.close()
        return False

    def start(self, env) -> None:
        """Start servers for ``env`` if their ports are free."""
        if env.combined_server:
            if not self._port_in_use(env.action_addr):
                cmd = [sys.executable, '-m', 'servers.action_server']
                p = subprocess.Popen(cmd)
                self._processes.append(p)
        else:
            if not self._port_in_use(env.reward_addr):
                cmd = [sys.executable, '-m', 'servers.reward_server']
                p = subprocess.Popen(cmd)
                self._processes.append(p)

        if env.use_world_model and not self._port_in_use(env.wm_addr):
            cmd = [
                sys.executable, '-m', 'servers.world_model_server',
                '--model-path', env.world_model_path,
                '--model-type', env.world_model_type,
                '--host', env.wm_addr[0],
                '--port', str(env.wm_addr[1])
            ]
            p = subprocess.Popen(cmd)
            self._processes.append(p)

    def stop(self) -> None:
        """Terminate all started server processes."""
        for proc in self._processes:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
        self._processes.clear()
