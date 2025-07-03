import gymnasium as gym
import numpy as np

class ContinuousBanditEnv(gym.Env):
    """Simple continuous action environment with no episode termination."""
    metadata = {"render_modes": []}

    def __init__(self, optimal_action: float = 0.5):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.state = np.zeros(1, dtype=np.float32)
        self.optimal_action = float(optimal_action)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.state.fill(0.0)
        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        reward = -abs(float(action[0]) - self.optimal_action)
        self.state[0] = reward
        return self.state.copy(), float(reward), False, False, {}

    def render(self):
        pass

    def close(self):
        pass
