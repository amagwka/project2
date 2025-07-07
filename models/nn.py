import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim=384, action_dim=7, net_width=256, max_action=1.0, lstm_hidden=256):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, lstm_hidden,num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden, net_width)
        self.fc2 = nn.Linear(net_width, action_dim)
        self.max_action = max_action

    def forward(self, state_seq):
        assert isinstance(state_seq, torch.Tensor), "state_seq must be a torch.Tensor"
        assert state_seq.dim() == 3, f"state_seq must be 3D [batch_size, seq_len, state_dim], got {state_seq.shape}"

        lstm_out, _ = self.lstm(state_seq)           # (batch, seq_len, lstm_hidden)
        x = lstm_out[:, -1]                          # use last output in sequence (batch, lstm_hidden)
        x = torch.relu(self.fc1(x))
        x = self.max_action * torch.tanh(self.fc2(x))
        return x

class Q_Critic(nn.Module):
    def __init__(self, shared_lstm, action_dim=7, net_width=256):
        super().__init__()
        self.lstm = shared_lstm
        self.lstm_hidden = self.lstm.hidden_size

        self.fc1 = nn.Linear(self.lstm_hidden + action_dim, net_width)
        self.fc2 = nn.Linear(net_width, 1)

    def forward(self, state_seq, action):
        assert isinstance(state_seq, torch.Tensor), "state_seq must be a torch.Tensor"
        assert state_seq.dim() == 3, f"state_seq must be 3D [batch_size, seq_len, state_dim], got {state_seq.shape}"
        batch_size, seq_len, state_dim = state_seq.shape

        # LSTM forward
        lstm_out , _ = self.lstm(state_seq)  # output: [batch_size, seq_len, lstm_hidden]
        assert lstm_out.shape[:2] == (batch_size, seq_len), "LSTM output shape mismatch"

        x = lstm_out[:, -1, :]  # last time step output → [batch_size, lstm_hidden]
        assert x.shape == (batch_size, self.lstm_hidden), f"x shape mismatch, got {x.shape}"

        # Action 
        assert isinstance(action, torch.Tensor), "action must be a torch.Tensor"

        assert action.shape[0] == batch_size, \
            f"action and state_seq batch size mismatch: {action.shape[0]} vs {batch_size}"
        #print(action.shape)
        #print(x.shape)
        x = torch.cat([x, action], dim=1)  # [batch_size, lstm_hidden + action_dim]
        expected_concat_dim = self.lstm_hidden + action.shape[1]
        assert x.shape == (batch_size, expected_concat_dim), \
            f"Concatenated input shape mismatch, got {x.shape}"
        #print(x.shape)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        assert x.shape == (batch_size, 1), f"Output shape mismatch, expected ({batch_size}, 1), got {x.shape}"

        return x.squeeze(1)  # Return shape: [batch_size]
