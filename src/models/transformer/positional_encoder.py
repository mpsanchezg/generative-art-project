import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_ep_len: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embed_timestep = nn.Embedding(max_ep_len, d_model)

    def forward(self, state, timestep):
        x = state + self.embed_timestep(timestep)
        return self.dropout(x)
