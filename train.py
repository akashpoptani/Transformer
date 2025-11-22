# train.py

import torch
from torch.utils.data import DataLoader

from model import Transformer
from dataset import ToyCopyDataset

# ============================
# Configuration variables
# ============================

VOCAB_SIZE = 100       # number of unique tokens
SEQ_LENGTH = 10        # length of each sequence
BATCH_SIZE = 32        # batch size
DATASET_SIZE = 5000    # number of sequences

#training
LEARNING_RATE = 1e-4
EPOCHS        = 20

# Model
D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

# ============================
# Dataset and DataLoader
# ============================
dataset = ToyCopyDataset(
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LENGTH,
    size=DATASET_SIZE                     # can make this a variable if you want
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================
# Create the model
# (must match vocab size)
# ============================
model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# ============================
# Training Loop
# ============================

for epoch in range(EPOCHS):
    for src, tgt in loader:

        optimizer.zero_grad()

        # tgt[:, :-1] = all tokens except the last → input to decoder
        output = model(src, tgt[:, :-1])
        # print(src, tgt, output)

        # Reshape outputs and targets for CrossEntropy:
        #   output: (batch, seq_len-1, vocab_size)
        #   tgt[:, 1:]: the "next tokens"
        loss = loss_fn(
            output.reshape(-1, VOCAB_SIZE),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss {loss.item():.4f}")
