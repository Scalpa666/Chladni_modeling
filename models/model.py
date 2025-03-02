import torch
import torch.nn as nn

# Define neural network for modeling acoustic displacement field
class AcDispNetL4(nn.Module):
    def __init__(self, n_input, n_output):
        super(AcDispNetL4, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_input, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_output)
        )

    def forward(self, x):
        return self.seq(x)


class AcDispNetL8(nn.Module):
    def __init__(self, n_input, n_output):
        super(AcDispNetL8, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_input, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_output)
        )

    def forward(self, x):
        return self.seq(x)