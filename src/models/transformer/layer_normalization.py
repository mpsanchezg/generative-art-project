import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**(-6)):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # No changes, had initially thought -2 since I thought seq_len would be second to last,
        # be after embedding it should be last aswell right?
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        
        return self.alpha * (x - mean) / ( std + self.eps) + self.bias
