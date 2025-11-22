# model/layers.py
import torch.nn as nn
from .attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ff(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        # Masked self-attention
        attn1 = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + attn1)

        # Cross-attention (decoder queries, encoder K,V)
        attn2 = self.cross_attn(x, enc_out, enc_out, cross_mask)
        x = self.norm2(x + attn2)

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        return x