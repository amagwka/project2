import gymnasium as gym
import numpy as np
import importlib
import threading
from time import sleep, perf_counter
from servers.manager import ServerManager
from typing import Optional, Callable
from config import EnvConfig, get_config

# Load default action-related constants from the global configuration
_DEFAULT_CFG = get_config().action_server
ARROW_DELAY = _DEFAULT_CFG.arrow_delay
WAIT_DELAY = _DEFAULT_CFG.wait_delay
NON_ARROW_DELAY = _DEFAULT_CFG.non_arrow_delay
ARROW_IDX = _DEFAULT_CFG.arrow_idx
WAIT_IDX = _DEFAULT_CFG.wait_idx

from utils.observation_encoder import ObservationEncoder
from utils.intrinsic import E3BIntrinsicReward, BaseIntrinsicReward
from utils.cosine import cosine_distance
from utils.nats_client import (
    NatsActionClient,
    NatsWorldModelClient,
    NatsIntrinsicClient,
)
from dataclasses import asdict

class NatsAppEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps=1000,
        device="cuda",
        action_dim=7,
        state_dim=384,
        embedding_model="facebook/dinov2-with-registers-small",
        start_servers=True,
        enable_logging=True,
        use_world_model=True,
        nats_url="nats://127.0.0.1:4222",
        world_model_path="lab/scripts/mlp_world_model.pt",
        world_model_type="mlp",
        world_model_interval=5,
        use_intrinsic_server=False,
        intrinsic_reward_name="E3BIntrinsicReward",
        config: Optional[EnvConfig] = None,
        action_client: Optional[NatsActionClient] = None,
        world_model_client: Optional[NatsWorldModelClient] = None,
        intrinsic_client: Optional[NatsIntrinsicClient] = None,
        obs_encoder: Optional[ObservationEncoder] = None,
        intrinsic_reward: Optional[BaseIntrinsicReward] = None,
        intrinsic_rewards: Optional[list[BaseIntrinsicReward]] = None,
        intrinsic_names: Optional[list[str]] = None,
        server_launcher: Optional[Callable[["NatsAppEnv"], None]] = None,
        server_manager: Optional[ServerManager] = None,
    ):

        super().__init__()

        self.max_steps = max_steps
        self.device = device
        self.step_count = 0
        self.last_action: int | None = None
        self.action_dim = action_dim
        self.state_dim = state_dim

        # NATS configuration
        self.embedding_model = embedding_model
        self.start_servers = start_servers
        self.enable_logging = enable_logging
        self.use_world_model = use_world_model
        self.nats_url = nats_url
        self.world_model_path = world_model_path
        self.world_model_type = world_model_type
        self.wm_interval_steps = int(max(1, world_model_interval))
        self.use_intrinsic_server = use_intrinsic_server
        self.intrinsic_reward_name = intrinsic_reward_name

        # Override above attributes from the config if provided
        cfg_names = self._init_from_config(config)
        if intrinsic_names is None:
            intrinsic_names = cfg_names

        self.action_space = gym.spaces.Discrete(self.action_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self._server_manager = server_manager
        self._logger = None
        self._last_action_time = perf_counter()

        self._init_clients(action_client, world_model_client, intrinsic_client)
        self.obs_history = []

        self.obs_encoder = obs_encoder or ObservationEncoder(
            source=1,
            model_name=self.embedding_model,
            device=self.device,
            embedding_dim=self.state_dim,
        )
        self._init_intrinsic(intrinsic_reward, intrinsic_rewards, intrinsic_names)

        if self.enable_logging:
            from utils import logger
            self._logger = logger

        self._server_launcher = server_launcher
        self._server_thread = None

        if self.start_servers:
            if self._server_launcher is not None:
                t = threading.Thread(target=self._server_launcher, args=(self,), daemon=True)
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
        self.last_action = None
        self._send_reset()
        if self.use_intrinsic_server and self.intrinsic_client is not None:
            self.intrinsic_client.send_reset()
        for ir in self.intrinsic_rewards:
            ir.reset()
        self.obs_history.clear()
        emb_np = self.obs_encoder.get_embedding()
        return emb_np.astype(np.float32), {}

    def step(self, action):
        self.step_count += 1
        self.last_action = action
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
        if self.use_intrinsic_server and self.intrinsic_client is not None:
            try:
                intrinsic = self.intrinsic_client.compute(obs_np, action)
            except Exception:
                intrinsic = 0.0
        else:
            intrinsic = float(
                sum(ir.compute(obs_np, self) for ir in self.intrinsic_rewards)
            )
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
        self.action_client.send_action(action_idx)

    def _get_reward(self):
        return self.action_client.get_reward()

    def _send_reset(self):
        self.action_client.send_reset()

    def _init_from_config(self, config: Optional[EnvConfig]):
        """Populate attributes from a configuration object."""
        if config is None:
            return []

        self.max_steps = config.max_steps
        self.device = config.device
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.embedding_model = config.embedding_model
        self.start_servers = config.start_servers
        self.enable_logging = config.enable_logging
        self.use_world_model = config.use_world_model
        self.world_model_path = config.world_model.model_path
        self.world_model_type = config.world_model.model_type
        self.wm_interval_steps = int(max(1, config.world_model.interval_steps))
        self.use_intrinsic_server = getattr(config, "use_intrinsic_server", False)
        self.nats_url = getattr(config, "nats_url", "nats://127.0.0.1:4222")
        if hasattr(config, "intrinsic_server"):
            self.intrinsic_reward_name = config.intrinsic_server.reward_name

        names = getattr(config, "intrinsic_names", None)
        return list(names) if names is not None else []

    def _init_clients(
        self,
        action_client: Optional[NatsActionClient],
        world_model_client: Optional[NatsWorldModelClient],
        intrinsic_client: Optional[NatsIntrinsicClient] = None,
    ) -> None:
        """Initialize NATS clients used by the environment."""
        self.action_client = action_client or NatsActionClient(self.nats_url)
        if self.use_world_model:
            self.wm_client = world_model_client or NatsWorldModelClient(self.nats_url)
        else:
            self.wm_client = None

        if self.use_intrinsic_server:
            self.intrinsic_client = intrinsic_client or NatsIntrinsicClient(self.nats_url)
        else:
            self.intrinsic_client = None

    def _instantiate_intrinsic(self, cls):
        """Instantiate ``cls`` with ``latent_dim``/``device`` if possible."""

        try:
            return cls(latent_dim=self.state_dim, device=self.device)
        except TypeError:
            return cls()

    def _init_intrinsic(
        self,
        intrinsic_reward: Optional[BaseIntrinsicReward],
        intrinsic_rewards: Optional[list[BaseIntrinsicReward]],
        intrinsic_names: list[str],
    ) -> None:
        """Instantiate the intrinsic reward helper."""
        self.intrinsic_rewards: list[BaseIntrinsicReward] = []

        if self.use_intrinsic_server:
            return

        if intrinsic_reward is not None:
            self.intrinsic_rewards.append(intrinsic_reward)

        if intrinsic_rewards:
            self.intrinsic_rewards.extend(intrinsic_rewards)

        for name in intrinsic_names:
            cls = None
            try:
                from utils.intrinsic_registry import get_reward
                cls = get_reward(name)
            except KeyError:
                cls = None

            if cls is None and "." in name:
                try:
                    mod_name, cls_name = name.rsplit(".", 1)
                    module = importlib.import_module(mod_name)
                    cls = getattr(module, cls_name)
                except Exception:
                    cls = None

            if cls is None:
                raise ValueError(f"Could not load intrinsic reward class '{name}'")

            self.intrinsic_rewards.append(self._instantiate_intrinsic(cls))

        if not self.intrinsic_rewards:
            self.intrinsic_rewards.append(
                E3BIntrinsicReward(
                    latent_dim=self.state_dim,
                    decay=1.0,
                    ridge=0.1,
                    device=self.device,
                )
            )

    def render(self):
        pass  # No GUI

    def close(self):
        self.action_client.close()
        if self.wm_client is not None:
            self.wm_client.close()
        if self.intrinsic_client is not None:
            self.intrinsic_client.close()
        if hasattr(self.obs_encoder, "close"):
            self.obs_encoder.close()
        if self._server_manager is not None:
            self._server_manager.stop()
        if self._server_thread is not None:
            self._server_thread.join(timeout=1)


def create_nats_env(cfg: EnvConfig, server_manager: Optional[ServerManager] = None) -> NatsAppEnv:
    """Instantiate ``NatsAppEnv`` from an ``EnvConfig``."""
    env_kwargs = asdict(cfg)
    wm = env_kwargs.pop("world_model")
    intr = env_kwargs.pop("intrinsic_server")
    env_kwargs.update({
        "world_model_path": wm["model_path"],
        "world_model_type": wm["model_type"],
        "world_model_interval": wm["interval_steps"],
        "intrinsic_reward_name": intr["reward_name"],
    })
    env_kwargs.pop("intrinsic_names", None)
    env_kwargs["config"] = cfg
    env_kwargs["server_manager"] = server_manager
    return NatsAppEnv(**env_kwargs)
