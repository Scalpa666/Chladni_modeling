import torch
import torch.nn as nn

# Define neural network for modeling acoustic displacement field
class AcDispNet(nn.Module):
    def __init__(self, n_input, n_output):
        super(AcDispNet, self).__init__()
        self.input = nn.Linear(n_input, 16)
        self.hidden = nn.Linear(16, 64)
        self.hidden2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, n_output)

    def forward(self, x):
        x = self.input(x)
        x = torch.relu(x)
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.out(x)
        return x
