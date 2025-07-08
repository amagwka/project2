# Project

This repository contains various components for interacting with an external application via UDP.

All default parameters for servers, training and the environment are
centralized in `config.py`.  Import `get_config()` to obtain a fresh
configuration object and pass its sections to the relevant modules.

## Environment

`SocketAppEnv` in `envs/socket_env.py` provides an OpenAI Gym environment. By default it communicates with two separate UDP ports: one for sending actions and one for requesting rewards. When the environment is created with `combined_server=True`, both actions and reward requests are sent to the same address. This mode is intended for use with the `start_combined_udp_server` helper defined in `servers/action_server.py`.

Example usage with the new helper classes:

```python
from utils.udp_client import UdpClient
from utils.observation_encoder import ObservationEncoder
from envs.socket_env import SocketAppEnv

with UdpClient(("127.0.0.1", 5005), ("127.0.0.1", 5006)) as udp:
    encoder = ObservationEncoder(device="cuda")
    env = SocketAppEnv(udp_client=udp, obs_encoder=encoder, start_servers=False)
```

# Undertale RL via UDP

This repository implements a reinforcement learning setup that interacts with an external application (e.g. Undertale) over UDP sockets. The main environment is `SocketAppEnv` in `envs/socket_env.py` which sends discrete actions and requests rewards from separate UDP endpoints. Observations are video frames encoded through a DINO model and combined with an intrinsic reward from `E3BIntrinsicReward` by default. Any module implementing the `BaseIntrinsicReward` interface can be passed via the `intrinsic_reward` parameter or loaded via the configuration for custom curiosity bonuses.

## Environment Workflow
* Actions are sent over UDP to a keyboard server.
* Extrinsic rewards are fetched from another UDP port.
* Observations are encoded with `LocalObs` and passed into an intrinsic reward module for novelty bonuses (defaults to `E3BIntrinsicReward`).
* Extrinsic and intrinsic rewards are summed for each step.

## Custom Curiosity Modules
The environment can dynamically load a curiosity plugin via the
``EnvConfig.intrinsic_name`` setting. Each plugin must implement the
``BaseIntrinsicReward`` interface which defines ``reset()`` and ``compute(obs, env)``.

Minimal example implementing a constant bonus:

```python
from utils.intrinsic import BaseIntrinsicReward

class MyReward(BaseIntrinsicReward):
    def reset(self) -> None:
        pass

    def compute(self, observation, env) -> float:
        return 1.0

# Direct instantiation
env = SocketAppEnv(intrinsic_reward=MyReward(), start_servers=False)

# Or via configuration
from config import get_config
cfg = get_config()
cfg.env.intrinsic_name = "examples.custom_curiosity.MyReward"
env = SocketAppEnv(config=cfg.env, start_servers=False)
```

See ``examples/custom_curiosity.py`` for a complete module returning a constant
bonus.

## Action and Reward Server

Both action and reward helpers are built on top of `UdpServer` defined in
`servers/base.py`.  `start_combined_udp_server` from `servers/action_server.py`
listens on a single port and dispatches incoming messages to either the keyboard
handler or the reward tracker.  `start_udp_reward_server` in
`servers/reward_server.py` exposes only the reward commands.  Each handler
understands the `GET` and `RESET` commands and replies with the computed reward
or ``OK`` respectively.  Run ``python -m servers.action_server`` to launch the
combined server.

`servers/action_server.py` exposes `start_combined_udp_server` which accepts any
`RewardTracker` implementation. The default tracker is `ExternalRewardTracker`
from `servers/reward_server.py`. Custom plugins can subclass
`RewardTracker`—see `examples/constant_tracker.py` for a trivial tracker. The
server now ignores `ConnectionResetError` events so occasional UDP connection
resets on Windows do not terminate the process. Run this module to start the
server.


## Neural Modules and PPO
* `models/nn.py` contains LSTM based actor and critic networks.
* `models/ppo.py` performs the PPO update using log‑prob ratios and clipping.
* `utils/rollout.py` provides a rollout buffer and GAE computation.

## Setup
Install the Python dependencies with the helper script:

```bash
./scripts/setup.sh
```

The script creates a virtual environment in `.venv` and installs the
packages pinned in `requirements.txt`.

## Running
1. Start the combined UDP server:
   ```bash
   python servers/action_server.py
   ```
2. (Optional) start the world model server which provides predictions for
   the additional reward. The server loads the pretrained checkpoint at
   `lab/scripts/rnn_lstm.pt` which was trained with an LSTM of hidden size
   512 and 3 layers:

   Note that observations produced by `LocalObs` are scaled down by a
   factor of ten. The environment scales them back up before sending them to
   the world model server, which in turn divides its predictions by ten
   before replying.

   ```bash
   python servers/world_model_server.py
   ```
3. Launch the training script:
   ```bash
   python main.py [--sb3] [--timesteps N]
   ```
   When running without `--sb3`, press **F9** to pause or resume training.
4. Run a quick random-action demo to inspect rewards:
   ```bash
   python examples/random_agent.py
   ```

Dependencies include `torch`, `gymnasium`, `numpy`, `transformers`, `pymem`,
`psutil`, `pynput` and `opencv-python`.

### Stable Baselines Example

To train using the PPO implementation from `stable-baselines3` on any of the
provided environments, run:

```bash
python examples/stable_ppo.py --env continuous  # bandit|continuous|socket
```

The script supports `MultiArmedBanditEnv`, the continuous
`ContinuousBanditEnv` and the main `SocketAppEnv` used for Undertale RL.
It demonstrates how to plug Stable Baselines PPO into the existing
infrastructure.
