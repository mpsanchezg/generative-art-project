import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, action_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, action_size)

    def forward(self, x):
        return torch.tanh(self.projection(x))
