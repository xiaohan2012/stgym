import torch_geometric as pyg
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


def get_pooling_function(name):
    if name == "mincut":
        return pyg.nn.dense.mincut_pool
    elif name == "sum":
        return global_add_pool
    elif name == "mean":
        return global_mean_pool
    elif name == "max":
        return global_max_pool
    elif name == "dmod":
        return pyg.nn.dense.dmon_pool
    else:
        raise ValueError
