import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from utils.rollout import RolloutBufferNoDone


def test_get_latest_state_seq_basic():
    buf = RolloutBufferNoDone(size=10, state_dim=2, action_dim=1, device="cpu")
    # Add a few states
    for i in range(3):
        s = torch.tensor([float(i), float(i)])
        a = torch.zeros(1)
        buf.add(s, a, 0.0, 0.0, 0.0)

    new_state = torch.tensor([99.0, 99.0])
    seq = buf.get_latest_state_seq(new_state)

    assert seq.shape == (1, buf.seq_len, 2)
    # First entries should be [new_state, last_state, second_last_state]
    assert torch.allclose(seq[0, 0], new_state)
    assert torch.allclose(seq[0, 1], torch.tensor([2.0, 2.0]))
    assert torch.allclose(seq[0, 2], torch.tensor([1.0, 1.0]))
