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
`servers/nats_base.py`. Individual modules are exposed over dedicated subjects:

* `servers/e3m_reward_server.py` publishes intrinsic rewards on
  ``rewards.e3m``.
* `servers/world_model_reward_server.py` returns world model errors on
  ``rewards.world_model``.
* `servers/external_reward_server.py` provides in‑game rewards via
  ``rewards.in_game``.
* `servers/action_executor_server.py` listens on ``actions.*`` queues to
  execute keyboard actions.

All servers understand the ``RESET`` command where applicable. They can be
launched individually or orchestrated through ``servers/orchestrator.py``.


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
1. Start the NATS server and whichever modules you need. For example the action
   executor and in‑game reward server:
   ```bash
   nats-server &
   python -m servers.action_executor_server
   python -m servers.external_reward_server
   ```
2. (Optional) start the world model reward server which loads the pretrained
   checkpoint at `lab/scripts/rnn_lstm.pt`:

   Note that observations produced by `LocalObs` are scaled down by a
   factor of ten. The environment scales them back up before sending them to
   the world model server, which in turn divides its predictions by ten
   before replying.

 ```bash
  python -m servers.world_model_reward_server --model-path lab/scripts/rnn_lstm.pt
  ```
3. (Optional) launch the E3M intrinsic reward module:
   ```bash
   python -m servers.e3m_reward_server
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
