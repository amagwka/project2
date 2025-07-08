# Central configuration for the project.
# Contains all tunable parameters used across modules.

from dataclasses import dataclass, field
from typing import Dict, Set, List
import torch


@dataclass
class ActionServerConfig:
    """Parameters for the action/reward UDP server."""
    arrow_delay: float = 0.07
    wait_delay: float = 0.07 / 2
    non_arrow_delay: float = 0.03
    arrow_idx: Set[int] = field(default_factory=lambda: {0, 1, 2, 3})
    wait_idx: int = 6


@dataclass
class WorldModelConfig:
    """Settings related to the optional world model."""
    host: str = "127.0.0.1"
    port: int = 5007
    model_path: str = "lab/scripts/rnn_lstm.pt"
    model_type: str = "lstm"  # "mlp", "gru" or "lstm"
    interval_steps: int = 15


@dataclass
class EnvConfig:
    """Default parameters for ``SocketAppEnv``."""
    max_steps: int = int(1e10)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    action_dim: int = 7
    state_dim: int = 384
    action_host: str = "127.0.0.1"
    action_port: int = 5005
    reward_host: str = "127.0.0.1"
    reward_port: int = 5006
    embedding_model: str = "facebook/dinov2-with-registers-small"
    combined_server: bool = True
    start_servers: bool = True
    enable_logging: bool = True
    use_world_model: bool = True
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    intrinsic_names: list[str] = field(default_factory=lambda: ["E3BIntrinsicReward"])


@dataclass
class TrainingConfig:
    """Hyperparameters for PPO training."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim: int = 384
    action_dim: int = 7
    seq_len: int = 70
    rollout_len: int = 512
    update_every: int = 256
    learning_rate: float = 1e-4
    clip_eps: float = 0.2


@dataclass
class RewardServerConfig:
    """Memory offsets and reward weights for the external tracker."""
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "Experience Points": 0.1,
            "Health Points": 1.0,
            "Fight Metric": 0.1,
        }
    )


@dataclass
class Config:
    """Top level configuration object."""
    action_server: ActionServerConfig = field(default_factory=ActionServerConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward_server: RewardServerConfig = field(default_factory=RewardServerConfig)
    world_model_ckpt: str = "lab/scripts/rnn_lstm.pt"


def get_config() -> Config:
    """Return a fresh ``Config`` populated with defaults."""
    return Config()
