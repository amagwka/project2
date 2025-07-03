import gymnasium as gym
import numpy as np

class MultiArmedBanditEnv(gym.Env):
    """Simple multi-armed bandit environment for quick PPO tests."""
    metadata = {"render_modes": []}

    def __init__(self, probs=None, seed: int = 0):
        super().__init__()
        if probs is None:
            probs = [0.2, 0.5, 0.8]
        self.probs = np.array(probs, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(self.probs)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        # Cast ``action_space.n`` to ``int`` so downstream code using
        # ``one_hot`` with ``action_dim`` works without type errors
        self.action_space.n = int(self.action_space.n)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.state = np.zeros(1, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state.fill(0.0)
        return self.state.copy(), {}

    def step(self, action):
        p = self.probs[int(action)]
        reward = 1.0 if self.rng.random() < p else 0.0
        self.state[0] = reward
        return self.state.copy(), reward, False, False, {}

    def render(self):
        pass

    def close(self):
        pass
