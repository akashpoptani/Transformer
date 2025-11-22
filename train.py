# train.py

import torch
from torch.utils.data import DataLoader

from model import Transformer
from dataset import ToyCopyDataset

# ============================
# Configuration variables
# ============================

VOCAB_SIZE   = 3   # number of unique tokens in dataset + model
SEQ_LENGTH   = 2    # length of each input/output sequence
BATCH_SIZE   = 2    # number of samples processed together
LEARNING_RATE = 1e-4
EPOCHS        = 2

# ============================
# Dataset and DataLoader
# ============================
dataset = ToyCopyDataset(
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LENGTH,
    size=10                     # can make this a variable if you want
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
model = Transformer(vocab_size=VOCAB_SIZE)

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
