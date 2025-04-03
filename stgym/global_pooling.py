from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from stgym.config_schema import GlobalPoolingType


def get_pooling_operator(type: GlobalPoolingType):
    if type == "add":
        return global_add_pool
    elif type == "mean":
        return global_mean_pool
    elif type == "max":
        return global_max_pool
    else:
        raise ValueError(type)
