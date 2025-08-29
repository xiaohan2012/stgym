import torch


def get_activation_function(name):
    if name == "prelu":
        return torch.nn.PReLU()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "swish":
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function '{name}'")
