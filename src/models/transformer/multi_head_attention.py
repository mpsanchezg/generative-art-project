import torch
import torch.nn as nn

from math import sqrt

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model is not divisible by n_heads"

        self.d_keys = d_model // n_heads
        self.w_queries = nn.Linear(d_model, d_model)
        self.w_keys = nn.Linear(d_model, d_model)
        self.w_values = nn.Linear(d_model, d_model)

        self.w_outputs = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Module):
        d_keys = query.shape[-1]
        
        # (Batch, n_head, seq_len, d_k) -> # (Batch, n_head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / sqrt(d_keys)

        if mask is not None:
            # Replace all values where mask is 0 with -inf for softmax
            attention_scores.masked_fill_(mask == 0, -1e9)

        # (Batch, n_head, seq_len)
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, query, key, value, mask):
        _query = self.w_queries(query)
        _key = self.w_keys(key)
        _value = self.w_values(value)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, n_head, d_k) -->  (Batch, n_head, seq_len, d_k)
        _query = _query.view(_query.shape[0], _query.shape[1], self.n_heads, self.d_keys).transpose(1, 2)
        _key = _key.view(_key.shape[0], _key.shape[1], self.n_heads, self.d_keys).transpose(1, 2)
        _value = _value.view(_value.shape[0], _value.shape[-1], self.n_heads, self.d_keys).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query=_query, key=_key, value=_value, mask=mask, dropout=self.dropout)

        # (Batch, n_head, seq_len, d_k) --> (Batch, seq_len, n_head, d_k) --> (Batch, seq_len, d_model) 
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) 
        return self.w_outputs(x)
