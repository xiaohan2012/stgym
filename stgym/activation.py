import torch


def get_activation_function(name, inplace=False):
    if name == "prelu":
        return torch.nn.PReLU()
    elif name == "relu":
        return torch.nn.ReLU(inplace=inplace)
    elif name == "swish":
        return torch.nn.SiLU(inplace=inplace)
    else:
        raise ValueError(f"Unknown activation function '{name}'")
