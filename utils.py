# utils.py

import torch

def save_model(model, path):
    """
    Save the model's learned parameters (weights) to a file.

    model.state_dict() returns a dictionary containing all learnable tensors.
    torch.save() writes that dictionary to disk so you can load it later.

    Example:
        save_model(model, "transformer.pt")
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Load previously saved weights into a model instance.

    torch.load() reads the state_dict from disk.
    model.load_state_dict() copies those weights into the model.

    NOTE:
    You must create the same model architecture before loading the weights.

    Example:
        model = Transformer(...)
        load_model(model, "transformer.pt")
    """
    model.load_state_dict(torch.load(path))
