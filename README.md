# Project

This repository contains various components for interacting with an external application via UDP.

## Environment

`SocketAppEnv` in `envs/socket_env.py` provides an OpenAI Gym environment. By default it communicates with two separate UDP ports: one for sending actions and one for requesting rewards. When the environment is created with `combined_server=True`, both actions and reward requests are sent to the same address. This mode is intended for use with the `start_combined_udp_server` helper defined in `servers/action_server.py`.

# Undertale RL via UDP

This repository implements a reinforcement learning setup that interacts with an external application (e.g. Undertale) over UDP sockets. The main environment is `SocketAppEnv` in `envs/socket_env.py` which sends discrete actions and requests rewards from separate UDP endpoints. Observations are video frames encoded through a DINO model and combined with an intrinsic reward from `E3BIntrinsicReward`.

## Environment Workflow
* Actions are sent over UDP to a keyboard server.
* Extrinsic rewards are fetched from another UDP port.
* Observations are encoded with `LocalObs` and passed into `E3BIntrinsicReward` for novelty bonuses.
* Extrinsic and intrinsic rewards are summed for each step.

## Action and Reward Server
`servers/action_server.py` exposes `start_combined_udp_server` which wraps `ExternalRewardTracker` from `servers/reward_server.py` and processes both actions and reward queries on the same UDP port. Run this module to start the server.

## Neural Modules and PPO
* `models/nn.py` contains LSTM based actor and critic networks.
* `models/ppo.py` performs the PPO update using log‑prob ratios and clipping.
* `utils/rollout.py` provides a rollout buffer and GAE computation.

## Redundant Parts
* `ExternalRewardTracker` exists in both `action_module.py` and `external_reward.py`.
* `simple.ipynb` repeats UDP utilities and environment logic that already live in the modules.

## Recommendations
1. Consolidate reward tracking and server utilities into a single module.
2. Replace the notebook with a proper training script that uses the environment, networks, PPO loop and rollout buffer.

## Running
1. Start the combined UDP server:
   ```bash
   python servers/action_server.py
   ```
2. (Optional) start the world model server which provides predictions for
   the additional reward. The server loads the pretrained checkpoint at
   `lab/scripts/rnn_lstm.pt` which was trained with an LSTM of hidden size
   512 and 3 layers:

   ```bash
   python servers/world_model_server.py
   ```
3. Launch the training script:
   ```bash
   python main.py
   ```
4. Run a quick random-action demo to inspect rewards:
   ```bash
   python examples/random_agent.py
   ```

Dependencies include `torch`, `gymnasium`, `numpy`, `transformers`, `pymem`, `psutil`, `pynput` and `opencv-python`.