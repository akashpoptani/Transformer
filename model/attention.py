# model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------
# 1. Scaled Dot-Product Attention
# -----------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_v)
        mask: (batch, 1, seq_q, seq_k)
        """
        d_k = Q.size(-1)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Masking (for causal self-attn)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers to produce Q,K,V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output linear
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        B, seq_q, _ = query.shape
        B, seq_k, _ = key.shape

        # Compute Q,K,V
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Split heads
        Q = Q.view(B, seq_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, seq_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, seq_k, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output = self.attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, seq_q, self.d_model)

        return self.W_O(attn_output)

def generate_subsequent_mask(size):
    """
    Creates a lower-triangular binary mask.
    size: target sequence length
    """
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask  # shape (1,1,size,size)
