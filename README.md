# Project

This repository contains various components for interacting with an external application via NATS.

All default parameters for servers, training and the environment are
centralized in `config.py`.  Import `get_config()` to obtain a fresh
configuration object and pass its sections to the relevant modules.

## Environment

`NatsAppEnv` in `envs/nats_env.py` provides an OpenAI Gym environment. All communication happens over NATS subjects instead of raw UDP sockets.

Example usage with the new helper classes:

```python
from utils.observation_encoder import ObservationEncoder
from envs.nats_env import NatsAppEnv

encoder = ObservationEncoder(device="cuda")
env = NatsAppEnv(obs_encoder=encoder, start_servers=False)
```

# Undertale RL via NATS

This repository implements a reinforcement learning setup that interacts with an external application (e.g. Undertale) using NATS for all communication. The main environment is `NatsAppEnv` in `envs/nats_env.py` which sends discrete actions and requests rewards via NATS subjects. Observations are video frames encoded through a DINO model and combined with an intrinsic reward from `E3BIntrinsicReward` by default. Any module implementing the `BaseIntrinsicReward` interface can be passed via the `intrinsic_reward` parameter or loaded via the configuration for custom curiosity bonuses.

## Environment Workflow
* Actions are sent over the `actions` NATS subject.
* Extrinsic rewards are fetched from the same subject.
* Observations are encoded with `LocalObs` and passed into an intrinsic reward module for novelty bonuses (defaults to `E3BIntrinsicReward`).
* Extrinsic and intrinsic rewards are summed for each step.

## Custom Curiosity Modules
The environment can dynamically load one or more curiosity plugins via the
``EnvConfig.intrinsic_names`` setting. Each plugin must implement the
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
env = NatsAppEnv(intrinsic_reward=MyReward(), start_servers=False)

# Or via configuration
from config import get_config
cfg = get_config()
cfg.env.intrinsic_names = ["examples.custom_curiosity.MyReward"]
env = NatsAppEnv(config=cfg.env, start_servers=False)
```

To combine multiple rewards simply list several names:

```python
cfg.env.intrinsic_names = [
    "examples.custom_curiosity.MyReward",
    "E3BIntrinsicReward",
]
```

See ``examples/custom_curiosity.py`` for a complete module returning a constant
bonus.

## Action and Reward Server

Action and reward helpers use the `NatsServer` base class defined in
`servers/nats_base.py`. The convenience function `start_nats_combined_server`
from `servers/action_server.py` listens on the `actions` subject and dispatches
incoming messages to either the keyboard handler or the reward tracker.
`servers/reward_server.py` exposes `start_nats_reward_server` for reward queries
only. Each handler understands the `GET` and `RESET` commands and replies with
the computed reward or ``OK`` respectively. Run ``python -m servers.action_server``
to launch the combined server.

`servers/intrinsic_server.py` exposes `start_nats_intrinsic_server` which hosts a
single curiosity module over NATS. Pass `--name` with any registered reward
class to run it as a standalone process.

`servers/action_server.py` accepts any `RewardTracker` implementation. The
default tracker is `ExternalRewardTracker` from `servers/reward_server.py`.
Custom plugins can subclass `RewardTracker`—see `examples/constant_tracker.py`
for a trivial tracker.


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
1. Start the NATS server and the combined action/reward server:
   ```bash
   nats-server &
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
3. (Optional) start the intrinsic reward server to offload curiosity
   computation:
   ```bash
   python -m servers.intrinsic_server --name E3BIntrinsicReward
   ```
4. Launch the training script:
   ```bash
   python main.py [--sb3] [--timesteps N]
   ```
   When running without `--sb3`, press **F9** to pause or resume training.
5. Run a quick random-action demo to inspect rewards:
   ```bash
   python examples/random_agent.py
   ```

Dependencies include `torch`, `gymnasium`, `numpy`, `transformers`, `pymem`,
`psutil`, `pynput` and `opencv-python`.

### Stable Baselines Example

To train using the PPO implementation from `stable-baselines3` on any of the
provided environments, run:

```bash
python examples/stable_ppo.py --env continuous  # bandit|continuous|nats
```

The script supports `MultiArmedBanditEnv`, the continuous
`ContinuousBanditEnv` and the main `NatsAppEnv` used for Undertale RL.
It demonstrates how to plug Stable Baselines PPO into the existing
infrastructure.
