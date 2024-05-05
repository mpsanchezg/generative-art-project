import torch
import torch.nn as nn

import math

class InputEmbeddings(nn.Module):
    def __init__(self, input_size: int, d_model: int):
        super().__init__()
        self.d_model: int = d_model
        self.embedding = nn.Linear(input_size, self.d_model)
        # self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # self.embedding(x)
