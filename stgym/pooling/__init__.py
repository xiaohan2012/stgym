# from torch_geometric.nn.dense import DMoNPooling
from .dmon import DMoNPoolingLayer  # noqa
from .mincut import MincutPoolingLayer  # noqa

# def get_pooling_function(name):
#     if name == "mincut":
#         return pyg.nn.dense.mincut_pool
#     elif name == "sum":
#         return global_add_pool
#     elif name == "mean":
#         return global_mean_pool
#     elif name == "max":
#         return global_max_pool
#     elif name == "dmod":
#         return pyg.nn.dense.dmon_pool
#     else:
#         raise ValueError


def get_pooling_class(name):
    if name == "dmon":
        return DMoNPoolingLayer
    elif name == "mincut":
        return MincutPoolingLayer
    else:
        raise NotImplementedError(name)
