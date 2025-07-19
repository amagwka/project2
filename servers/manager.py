import subprocess
import sys
from typing import List


class ServerManager:
    """Launch and stop reward/action/world model servers."""

    def __init__(self) -> None:
        self._processes: List[subprocess.Popen] = []


    def start(self, env) -> None:
        """Start servers for ``env`` if their ports are free."""
        if env.start_servers:
            cmd = [sys.executable, '-m', 'servers.action_server']
            p = subprocess.Popen(cmd)
            self._processes.append(p)

        if env.use_world_model:
            cmd = [
                sys.executable, '-m', 'servers.world_model_server',
                '--model-path', env.world_model_path,
                '--model-type', env.world_model_type,
            ]
            p = subprocess.Popen(cmd)
            self._processes.append(p)

        if env.use_intrinsic_server:
            cmd = [
                sys.executable, '-m', 'servers.intrinsic_server',
                '--name', env.intrinsic_reward_name,
                '--subject', 'intrinsic',
                '--latent-dim', str(env.state_dim),
                '--device', env.device,
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
