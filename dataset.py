# dataset.py

import torch
from torch.utils.data import Dataset

class ToyCopyDataset(Dataset):
    """
    A simple dataset where the model must learn to copy the input sequence.

    For each sample:
        src = [a, b, c, ...]
        tgt = [a, b, c, ...]

    This is very useful for debugging Transformer models.

    vocab_size : number of possible tokens the model can see (1..vocab_size)
    seq_len    : length of each sequence
    size       : how many samples total in the dataset
    """

    def __init__(self, vocab_size=3, seq_len=10, size=5):
        self.vocab = vocab_size
        self.len = seq_len
        self.size = size

    def __len__(self):
        """Return number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx):
        """
        Generate one random sequence of integers.

        torch.randint creates a tensor of random integers.
        Each item is a sequence, and since it's a copy task,
        both src and tgt are identical.
        """
        seq = torch.randint(1, self.vocab, (self.len,))
        return seq, seq   # (input, target)
