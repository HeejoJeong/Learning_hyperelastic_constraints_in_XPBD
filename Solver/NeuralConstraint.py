import torch
import torch.nn as nn


class NeuralFEMConstriant_Invariants(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_gelu_stack = nn.Sequential(
            nn.Linear(3, 64,bias=False),
            nn.GELU(),
            nn.Linear(64, 64,bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False),
        )
        self._initialize_weights()
    def forward(self, x):
        x = self.flatten(x)
        constr = self.linear_gelu_stack(x)
        return constr
    def _initialize_weights(self):
        for layer in self.linear_gelu_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                with torch.no_grad():
                    layer.weight.div_(layer.weight.norm(dim=1, keepdim=True))

class NeuralFEMConstriant_Invariants_pos_based(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_gelu_stack = nn.Sequential(
            nn.Linear(3, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self._initialize_weights()
    def forward(self, x):
        x = self.flatten(x)
        constr = self.linear_gelu_stack(x)
        return constr
    def _initialize_weights(self):
        for layer in self.linear_gelu_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                with torch.no_grad():
                    layer.weight.div_(layer.weight.norm(dim=1, keepdim=True))

