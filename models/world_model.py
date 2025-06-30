import torch
import torch.nn as nn

class LSTMWorldModel(nn.Module):
    """Simple LSTM world model predicting the next observation embedding."""

    def __init__(self, obs_dim: int = 384, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        # Saved checkpoints from ``lab/scripts/train_rnn.py`` use the module
        # name ``rnn`` for the recurrent layer.  Keep the same name here so the
        # provided ``rnn_lstm.pt`` loads without key mismatches.
        self.rnn = nn.LSTM(obs_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, obs_dim)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs_seq: Tensor of shape ``(batch, seq_len, obs_dim)`` containing the
                observation history.
        Returns:
            Tensor of shape ``(batch, obs_dim)`` representing the predicted next
            observation embedding.
        """
        out, _ = self.rnn(obs_seq)
        out = out[:, -1]
        return self.fc(out)
