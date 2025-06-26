# Project Overview

This repository contains experiments around training a reinforcement learning (RL) agent to interact with the game **Undertale**. The code is split across multiple small modules under `components/` and a Jupyter notebook (`simple.ipynb`) that demonstrates manual testing.

## Key Modules

| File | Purpose |
| ---- | ------- |
| `components/env.py` | Defines `SocketAppEnv`, a Gym environment that communicates with UDP servers for actions and rewards. Observations are image embeddings from `LocalObs` and an intrinsic reward from `E3BIntrinsicReward`. |
| `components/action_module.py` | Provides keyboard control logic and a combined UDP server for sending actions and requesting reward. Contains an `ExternalRewardTracker` that reads game memory via `pymem`. |
| `components/external_reward.py` | A standalone version of the reward tracker and UDP server. Functionally overlaps with the one in `action_module.py`. |
| `components/obs.py` | Implements `LocalObs` which captures webcam/screen frames and produces DINO embeddings. |
| `components/obs_get.py` | More advanced asynchronous frame collector with optional preview. Not used by the environment directly. |
| `components/socket_obs.py` | TCP server that streams raw frames or embeddings over a socket. Useful for remote observation but not referenced elsewhere. |
| `components/rew.py` | `E3BIntrinsicReward` – novelty measure computed from embeddings. |
| `components/rollout.py` | `RolloutBufferNoDone` for storing sequences of states/actions and utilities to compute GAE. |
| `components/nn.py` | Actor and critic network definitions using an LSTM backbone. |
| `components/ppo.py` | Single `ppo_update` function implementing the PPO update step. |
| `components/shrink.py` | `LogStencilMemory` class for down‑sampling stored embeddings. Appears unused. |

`simple.ipynb` ties many of these pieces together and contains assorted helper functions for manual testing.

## How Components Interact

1. **Action/Reward Servers** – `action_module.py` (or `external_reward.py`) launches UDP servers. One listens for action indices and simulates key presses. The other exposes game metrics as a numeric reward via memory scanning.
2. **Environment** – `SocketAppEnv` sends actions to the action server and queries the reward server. It also captures an observation using `LocalObs` and computes intrinsic reward using `E3BIntrinsicReward`.
3. **Training** – Networks from `nn.py` and utilities from `rollout.py` and `ppo.py` can be combined to train an agent that interacts with `SocketAppEnv`. The notebook contains initial experimentation with these pieces.

```
[Agent] -> env.step(action) -> [SocketAppEnv]
         -> UDP action server (keyboard)
         <- observation embedding
         <- UDP reward server (memory read)
```

## Redundant or Unused Code

* `action_module.py` and `external_reward.py` both implement nearly identical reward trackers/servers.
* `socket_obs.py` provides a frame server that is not referenced in other modules.
* `obs_get.py` and `shrink.py` are utilities that are not hooked into `SocketAppEnv`.
* Several files contain leftover shell prompts or truncated lines (e.g. at the end of `action_module.py`, `rollout.py`, and others), indicating accidental copy‑paste during development.
* `simple.ipynb` duplicates a lot of logic from the modules and mixes exploratory code with utility functions.

## Suggested Improvements

* Convert `simple.ipynb` into a script (e.g. `main.py`) so the project can be run without Jupyter.
* Clean up the corrupted lines in `components/action_module.py`, `components/rollout.py`, and similar files.
* Consolidate the reward tracker into a single implementation to avoid confusion.
* Consider reorganizing `components/` into packages such as `env/`, `networks/`, and `utils/` for clarity.
* The socket environment could be extended to allow configurable ports and timeouts via constructor arguments rather than constants scattered in the code.

## Open Tasks

- [ ] **Create `main.py`** that demonstrates running `SocketAppEnv` with the PPO loop.
- [ ] **Fix truncated code** in `components/action_module.py` and `components/rollout.py`.
- [ ] **Refactor `SocketAppEnv`** so the action and reward endpoints are easier to mock or swap out (e.g. dependency injection of sockets).
- [ ] **Remove unused modules** or move experimental utilities under a dedicated `examples/` folder.

The project is licensed under the MIT License (see [LICENSE](LICENSE)).
