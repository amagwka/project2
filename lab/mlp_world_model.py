import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPWorldModel(nn.Module):
    """Predicts the next embedding from the current embedding."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 512, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mlp_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combination of MSE and cosine similarity loss used for prediction."""
    mse = F.mse_loss(prediction, target)
    cosine = 1 - F.cosine_similarity(prediction, target).mean()
    return mse + cosine
