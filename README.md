# Undertale RL via UDP

This repository implements a reinforcement learning setup that interacts with an external application (e.g. Undertale) over UDP sockets. The main environment is `SocketAppEnv` in `components/env.py` which sends discrete actions and requests rewards from separate UDP endpoints. Observations are video frames encoded through a DINO model and combined with an intrinsic reward from `E3BIntrinsicReward`.

## Environment Workflow
* Actions are sent over UDP to a keyboard server.
* Extrinsic rewards are fetched from another UDP port.
* Observations are encoded with `LocalObs` and passed into `E3BIntrinsicReward` for novelty bonuses.
* Extrinsic and intrinsic rewards are summed for each step.

## Action and Reward Server
`components/action_module.py` exposes `start_combined_udp_server` which wraps `ExternalRewardTracker` from `external_reward.py` and processes both actions and reward queries on the same UDP port. Run this module to start the server.

## Neural Modules and PPO
* `components/nn.py` contains LSTM based actor and critic networks.
* `components/ppo.py` performs the PPO update using log‑prob ratios and clipping.
* `components/rollout.py` provides a rollout buffer and GAE computation.

## Redundant Parts
* `ExternalRewardTracker` exists in both `action_module.py` and `external_reward.py`.
* `simple.ipynb` repeats UDP utilities and environment logic that already live in the modules.

## Recommendations
1. Consolidate reward tracking and server utilities into a single module.
2. Replace the notebook with a proper training script that uses the environment, networks, PPO loop and rollout buffer.

## Running
1. Start the combined UDP server:
   ```bash
   python components/action_module.py
   ```
2. Launch the training script (not provided yet – create one using the modules above).

Dependencies include `torch`, `gymnasium`, `numpy`, `transformers`, `pymem`, `psutil`, `pynput` and `opencv-python`.
