# log_memory.py

import numpy as np

class LogStencilMemory:
    def __init__(self, max_len=1000, steps=70, feature_dim=384):
        self.max_len = max_len
        self.steps = steps
        self.feature_dim = feature_dim
        self.memory = np.zeros((0, feature_dim), dtype=np.float32)

    def add(self, element):
        """Add a (1, feature_dim) element. If over capacity, remove the oldest (first) row."""
        if element.shape != (1, self.feature_dim):
            raise ValueError(f"Element must have shape (1, {self.feature_dim})")
        self.memory = np.concatenate([self.memory, element], axis=0)
        if len(self.memory) > self.max_len:
            self.memory = self.memory[1:]  # Drop the oldest (first) element

    def shrinked(self):
        """Return a log-stenciled subset from the most recent (not oldest) elements."""
        mem_len = len(self.memory)
        if mem_len == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        lin_steps = np.round(np.linspace(1, max(2, mem_len // (self.steps//2)), self.steps)).astype(int)
        indices = np.cumsum(np.insert(lin_steps, 0, 0))[:-1]
        indices = indices[indices < mem_len]
        reversed_indices = mem_len - 1 - indices
        reversed_indices = reversed_indices[::-1]
        return self.memory[reversed_indices]


# Example usage:
if __name__ == "__main__":
    mem = LogStencilMemory(max_len=1000, steps=70, feature_dim=384)
    for i in range(1200):
        x = np.ones((1, 384), dtype=np.float32) * i
        mem.add(x)
        if i % 200 == 0:
            print(f"At step {i}, memory length: {len(mem)}")
            shrink = mem.shrinked()
            print(f"Shrinked memory shape: {shrink.shape}")
            print(f"Shrinked indices sample: {shrink[:,0][:10]}")  # Just to show index progression
