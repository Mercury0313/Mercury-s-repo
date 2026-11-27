"""Policy architecture from onepounchman/RACER, commit 9d80ecc."""

import torch
from torch import nn


class PolicyNet(nn.Module):
    """Return P(select candidate 1 | embedding), with shape [N, 1]."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))
