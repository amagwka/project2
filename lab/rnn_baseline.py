import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNPredictor(nn.Module):
    """Simple LSTM/GRU based predictor for next-frame embeddings."""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 512, num_layers: int = 2, rnn_type: str = "LSTM"):
        super().__init__()
        rnn_type = rnn_type.upper()
        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1]
        return self.fc(last)

def rnn_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    cosine = 1 - F.cosine_similarity(prediction, target).mean()
    return mse + cosine
