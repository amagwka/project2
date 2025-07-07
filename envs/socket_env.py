import gymnasium as gym
import numpy as np
import socket
import importlib
import threading
from time import sleep, perf_counter
from servers.constants import (
    ARROW_DELAY,
    WAIT_DELAY,
    NON_ARROW_DELAY,
    ARROW_IDX,
    WAIT_IDX,
)
from servers.manager import ServerManager
from servers import UdpServer
from typing import Optional, Callable
from config import EnvConfig

from utils.observation_encoder import ObservationEncoder
from utils.curiosity_base import IntrinsicReward
from utils.intrinsic import E3BIntrinsicReward, BaseIntrinsicReward
from utils.cosine import cosine_distance
from utils.udp_client import UdpClient
from utils.world_model_client import WorldModelClient
from dataclasses import asdict

class SocketAppEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps=1000,
        device="cuda",
        action_dim=7,
        state_dim=384,
        action_host="127.0.0.1",
        action_port=5005,
        reward_host="127.0.0.1",
        reward_port=5006,
        embedding_model="facebook/dinov2-with-registers-small",
        combined_server=True,
        start_servers=True,
        enable_logging=True,
        use_world_model=True,
        world_model_host="127.0.0.1",
        world_model_port=5007,
        world_model_path="lab/scripts/mlp_world_model.pt",
        world_model_type="mlp",
        world_model_interval=5,
        config: Optional[EnvConfig] = None,
        udp_client: Optional[UdpClient] = None,
        world_model_client: Optional[WorldModelClient] = None,
        obs_encoder: Optional[ObservationEncoder] = None,
        intrinsic_reward: Optional[BaseIntrinsicReward] = None,
        server_launcher: Optional[Callable[["SocketAppEnv"], Optional[UdpServer]]] = None,
        server_manager: Optional[ServerManager] = None,
    ):

        super().__init__()

        self.max_steps = max_steps
        self.device = device
        self.step_count = 0
        self.action_dim = action_dim
        self.state_dim = state_dim

        # UDP configuration
        self.action_addr = (action_host, action_port)
        self.reward_addr = (reward_host, reward_port)
        self.embedding_model = embedding_model
        self.combined_server = combined_server
        self.start_servers = start_servers
        self.enable_logging = enable_logging
        self.use_world_model = use_world_model
        self.wm_addr = (world_model_host, world_model_port)
        self.world_model_path = world_model_path
        self.world_model_type = world_model_type
        self.wm_interval_steps = int(max(1, world_model_interval))

        # Override above attributes from the config if provided
        ir_config, intrinsic_cls_path = self._init_from_config(config)

        self.action_space = gym.spaces.Discrete(self.action_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self._server_manager = server_manager
        self._logger = None
        self._last_action_time = perf_counter()

        self._init_udp_clients(udp_client, world_model_client)
        self.obs_history = []

        self.obs_encoder = obs_encoder or ObservationEncoder(
            source=1,
            model_name=self.embedding_model,
            device=self.device,
            embedding_dim=self.state_dim,
        )
        self._init_intrinsic(intrinsic_reward, intrinsic_cls_path, ir_config)

        if self.enable_logging:
            from utils import logger
            self._logger = logger

        self._server_launcher = server_launcher
        self._server_instance: Optional[UdpServer] = None
        self._server_thread = None

        if self.start_servers:
            if self._server_launcher is not None:
                srv = self._server_launcher(self)
                if isinstance(srv, UdpServer):
                    self._server_instance = srv
                    t = threading.Thread(target=srv.serve, daemon=True)
                    t.start()
                    self._server_thread = t
            elif self._server_manager is not None:
                self._server_manager.start(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self._send_reset()
        self.intrinsic.reset()
        self.obs_history.clear()
        emb_np = self.obs_encoder.get_embedding()
        return emb_np.astype(np.float32), {}

    def step(self, action):
        self.step_count += 1
        now = perf_counter()
        elapsed = now - self._last_action_time
        delay = 0.0
        if action in ARROW_IDX:
            delay = ARROW_DELAY
        elif action == WAIT_IDX:
            delay = WAIT_DELAY
        else:
            delay = NON_ARROW_DELAY
        self._send_action(action)
        if delay > 0 and elapsed < delay:
            sleep(delay - elapsed)
        self._last_action_time = perf_counter()

        obs_np = self.obs_encoder.get_embedding()
        extrinsic = self._get_reward()
        intrinsic = self.intrinsic.compute(obs_np, self)
        model_bonus = 0.0
        self.obs_history.append(obs_np.copy())
        if len(self.obs_history) > 30:
            self.obs_history.pop(0)
        if (
            self.use_world_model and
            len(self.obs_history) == 30 and
            self.step_count % self.wm_interval_steps == 0
        ):
            context = np.stack(self.obs_history, axis=0).astype(np.float32) * 10.0
            try:
                pred = self.wm_client.predict(context)
                if pred.size == self.state_dim:
                    dist = cosine_distance(pred, obs_np)
                    model_bonus = -dist * 10
            except socket.timeout:
                model_bonus = 99.99
            except Exception:
                model_bonus = 99.99

        reward = extrinsic + intrinsic + model_bonus

        if self._logger is not None:
            self._logger.log_scalar("Reward/Extrinsic", extrinsic, self.step_count)
            self._logger.log_scalar("Reward/Intrinsic", intrinsic, self.step_count)
            self._logger.log_scalar("Reward/ModelBonus", model_bonus, self.step_count)
            self._logger.log_scalar("Reward/Total", reward, self.step_count)

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {
            "extrinsic": float(extrinsic),
            "intrinsic": float(intrinsic),
            "model_bonus": float(model_bonus),
        }
        return obs_np.astype(np.float32), reward, terminated, truncated, info

    def _send_action(self, action_idx):
        self.udp_client.send_action(action_idx)

    def _get_reward(self):
        return self.udp_client.get_reward()

    def _send_reset(self):
        self.udp_client.send_reset()

    def _init_from_config(self, config: Optional[EnvConfig]):
        """Populate attributes from a configuration object."""
        if config is None:
            return None, None

        self.max_steps = config.max_steps
        self.device = config.device
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.action_addr = (config.action_host, config.action_port)
        self.reward_addr = (config.reward_host, config.reward_port)
        self.embedding_model = config.embedding_model
        self.combined_server = config.combined_server
        self.start_servers = config.start_servers
        self.enable_logging = config.enable_logging
        self.use_world_model = config.use_world_model
        self.wm_addr = (config.world_model.host, config.world_model.port)
        self.world_model_path = config.world_model.model_path
        self.world_model_type = config.world_model.model_type
        self.wm_interval_steps = int(max(1, config.world_model.interval_steps))

        ir_cfg = getattr(config, "intrinsic_reward", None)
        ir_cls = getattr(config, "intrinsic_cls", None)
        return ir_cfg, ir_cls

    def _init_udp_clients(
        self,
        udp_client: Optional[UdpClient],
        world_model_client: Optional[WorldModelClient],
    ) -> None:
        """Initialize UDP clients used by the environment."""
        self.udp_client = udp_client or UdpClient(
            self.action_addr, self.reward_addr, self.combined_server
        )
        if self.use_world_model:
            self.wm_client = world_model_client or WorldModelClient(self.wm_addr)
        else:
            self.wm_client = None

    def _instantiate_intrinsic(self, cls):
        """Instantiate ``cls`` with ``latent_dim``/``device`` if possible."""

        try:
            return cls(latent_dim=self.state_dim, device=self.device)
        except TypeError:
            return cls()

    def _init_intrinsic(
        self,
        intrinsic_reward: Optional[BaseIntrinsicReward],
        intrinsic_cls_path: Optional[str],
        ir_config: Optional[IntrinsicReward],
    ) -> None:
        """Instantiate the intrinsic reward helper."""
        if intrinsic_reward is not None:
            self.intrinsic = intrinsic_reward
            return

        if intrinsic_cls_path is not None:
            mod_name, cls_name = intrinsic_cls_path.rsplit(".", 1)
            module = importlib.import_module(mod_name)
            cls = getattr(module, cls_name)
            self.intrinsic = self._instantiate_intrinsic(cls)
            return

        if ir_config is not None:
            module = importlib.import_module(ir_config.module_path)
            cls = getattr(module, ir_config.class_name)
            self.intrinsic = self._instantiate_intrinsic(cls)
            return

        self.intrinsic = E3BIntrinsicReward(
            latent_dim=self.state_dim,
            decay=1.0,
            ridge=0.1,
            device=self.device,
        )

    def render(self):
        pass  # No GUI

    def close(self):
        self.udp_client.close()
        if self.wm_client is not None:
            self.wm_client.close()
        if hasattr(self.obs_encoder, "close"):
            self.obs_encoder.close()
        if self._server_manager is not None:
            self._server_manager.stop()
        if self._server_instance is not None:
            self._server_instance.shutdown()
            if self._server_thread is not None:
                self._server_thread.join(timeout=1)


def create_socket_env(cfg: EnvConfig, server_manager: Optional[ServerManager] = None) -> SocketAppEnv:
    """Instantiate ``SocketAppEnv`` from an ``EnvConfig``."""
    env_kwargs = asdict(cfg)
    wm = env_kwargs.pop("world_model")
    env_kwargs.update({
        "world_model_host": wm["host"],
        "world_model_port": wm["port"],
        "world_model_path": wm["model_path"],
        "world_model_type": wm["model_type"],
        "world_model_interval": wm["interval_steps"],
    })
    env_kwargs.pop("intrinsic_reward", None)
    env_kwargs.pop("intrinsic_cls", None)
    env_kwargs["config"] = cfg
    env_kwargs["server_manager"] = server_manager
    return SocketAppEnv(**env_kwargs)
