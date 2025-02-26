import torch

from stgym.config_schema import PoolingConfig

# from torch_geometric.nn.dense import DMoNPooling
from .dmon import DMoNPooling  # noqa

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


class DMoNPoolingLayer(torch.nn.Module):
    def __init__(self, cfg: PoolingConfig, **kwargs):
        super().__init__()
        # one linear layer
        self.model = DMoNPooling(channels=[], k=[cfg.n_clusters], dropout=0.0)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.adj)
        return batch


def get_pooling_class(name):
    if name == "dmon":
        return DMoNPoolingLayer
    else:
        raise NotImplementedError(name)
