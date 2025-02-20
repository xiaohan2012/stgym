import torch


def get_activation_function(name):
    if name == "prelu":
        return torch.nn.PReLU()
    elif name == "relu":
        return torch.nn.ReLU()
