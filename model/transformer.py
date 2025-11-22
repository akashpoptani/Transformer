# model/transformer.py
import torch
import torch.nn as nn
from .layers import EncoderLayer, DecoderLayer
from .attention import generate_subsequent_mask

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(5000, d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_decoder_layers)
        ])

        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        B, src_len = src.shape
        B, tgt_len = tgt.shape

        # Positional encoding
        src = self.embed(src) + self.pos_emb(torch.arange(src_len).to(src.device))
        tgt = self.embed(tgt) + self.pos_emb(torch.arange(tgt_len).to(tgt.device))

        # Masks
        tgt_mask = generate_subsequent_mask(tgt_len).to(src.device)

        # ENCODER
        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        # DECODER
        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, self_mask=tgt_mask)

        # Output layer
        logits = self.output_linear(dec_out)  # shape: (B, tgt_len, vocab_size)
        return logits
