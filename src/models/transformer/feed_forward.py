import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_feed_fwd: int, dropout: float):
        super().__init__()
        
        self.first_linear = nn.Linear(d_model, d_feed_fwd)
        self.dropout = nn.Dropout(dropout)
        self.second_linear = nn.Linear(d_feed_fwd, d_model)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        x = self.first_linear(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return self.second_linear(x)
