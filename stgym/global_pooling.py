from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from stgym.config_schema import GlobalPoolingConfig


def get_global_pooling_operator(cfg: GlobalPoolingConfig):
    if cfg.type == "add":
        return global_add_pool
    elif cfg.type == "mean":
        return global_mean_pool
    elif cfg.type == "max":
        return global_max_pool
    else:
        raise ValueError(cfg.type)
