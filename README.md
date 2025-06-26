# Project

This repository contains various components for interacting with an external application via UDP.

## Environment

`SocketAppEnv` in `components/env.py` provides an OpenAI Gym environment. By default it communicates with two separate UDP ports: one for sending actions and one for requesting rewards. When the environment is created with `combined_server=True`, both actions and reward requests are sent to the same address. This mode is intended for use with the `start_combined_udp_server` helper defined in `components/action_module.py`.

