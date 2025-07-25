import torch
import numpy as np

class RolloutBufferNoDone:
    def __init__(self, size, state_dim, action_dim,
                 device,
                 padding_value=0.0):
        self.size     = size
        self.device   = device
        self.full     = False
        self.ptr      = 0
        self.pad_val  = padding_value

        self.state    = torch.zeros(size, state_dim, device=device)
        self.action   = torch.zeros(size, action_dim, device=device)
        self.reward   = torch.zeros(size, device=device)
        self.value    = torch.zeros(size, device=device)
        self.logprob  = torch.zeros(size, device=device)
        # Default offsets roughly follow a logarithmic spacing to capture a wide
        # history while keeping sequences short.  For small buffers, fall back to
        # a simple increasing range so that at least one valid sequence exists.
        default_offsets = np.array([
            0, 1, 2, 4, 6, 9, 12, 15, 19, 23,
            28, 33, 38, 44, 50, 56, 63, 70, 78, 86,
            94, 103, 112, 122, 132, 142, 153, 164, 176, 188,
            200, 213, 226, 240, 254, 268, 283, 298, 313, 329,
            345, 362, 379, 396, 414, 432, 451, 470, 489, 509,
            529, 550, 571, 592, 614, 636, 659, 682, 705, 729,
            753, 777, 802, 827, 853, 879, 905, 932, 959, 987
        ], dtype=np.int64)

        if size <= 1:
            offsets = np.array([0], dtype=np.int64)
        elif size <= len(default_offsets):
            offsets = np.arange(size, dtype=np.int64)
        else:
            offsets = default_offsets

        self.offsets = torch.as_tensor(offsets[offsets < size], dtype=torch.long, device=device)
        self.seq_len = len(self.offsets)

    def add(self, s, a, r, v, lp):
        self.state   [self.ptr] = s
        self.action  [self.ptr] = a
        self.reward  [self.ptr] = r
        self.value   [self.ptr] = v
        self.logprob [self.ptr] = lp

        self.ptr  = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def ready(self):
        return self.full

    def get(self):
        if not self.full:
            raise RuntimeError("Buffer not filled yet!")

        N       = self.size
        p       = self.ptr                 # next write → oldest element
        max_off = int(self.offsets.max())  # largest lookback

        idx = torch.arange(N, device=self.device)

        # 1) Mask out any base index whose history would wrap the circular boundary
        mask = (idx < p) | (idx > p + max_off-1)
        bases = idx[mask]  # physical buffer indices

        # 2) Sort bases so oldest comes first (chronological order)
        ages      = (bases - p) % N        # 0 = oldest, up to (#valid-1)=newest
        sort_idx  = ages.argsort()
        bases     = bases[sort_idx]

        # 3) Build sequence indices and gather
        seq_idx    = (bases.unsqueeze(1) - self.offsets) % N  # (batch, seq_len)
        states_seq = self.state[seq_idx]

        return (
            states_seq,
            self.action[bases],
            self.reward[bases],
            self.value[bases],
            self.logprob[bases],
        )
    def get_latest_state_seq(self, new_state: torch.Tensor) -> torch.Tensor:
        if new_state.shape != (self.state.shape[1],):
            raise ValueError(f"Expected new_state shape ({self.state.shape[1]},), got {tuple(new_state.shape)}")
    
        N       = self.size
        p       = self.ptr
        avail   = N if self.full else p

        # Clamp offsets so ``idx`` never references uninitialised elements.
        max_off = max(avail - 1, 0)
        idx = (p - self.offsets.clamp(max=max_off)) % N

        seq = self.state[idx]

        # Pad where history is insufficient
        pad_mask = self.offsets > avail - 1
        seq[pad_mask] = self.pad_val

        # Insert the provided ``new_state`` for the most recent step
        seq[self.offsets == 0] = new_state

        return seq.unsqueeze(0)

        
def compute_gae(reward, value, gamma=0.995, lam=0.99):
    """Compute Generalised Advantage Estimation (GAE).

    When called with empty tensors, this returns empty results instead of raising
    an ``IndexError``.
    """
    T = reward.size(0)
    returns   = torch.empty_like(reward)
    advantage = torch.empty_like(reward)
    if T == 0:
        return returns, advantage

    next_val  = value[-1]
    gae = 0.0
    for t in reversed(range(T)):
        delta = reward[t] + gamma * next_val - value[t]
        gae   = delta + gamma * lam * gae
        advantage[t] = gae
        returns[t]   = gae + value[t]
        next_val     = value[t]
    std = advantage.std(unbiased=False)
    if std > 1e-8:
        advantage = (advantage - advantage.mean()) / (std + 1e-8)
    else:  # avoid NaNs when variance is zero
        advantage = advantage - advantage.mean()
    return returns, advantage
