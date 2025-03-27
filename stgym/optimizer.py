from typing import Iterator

import pydash as _
from torch.nn import Parameter
from torch.optim import SGD, Adam

from stgym.config_schema import OptimizerConfig


def create_optimizer(cfg: OptimizerConfig, params: Iterator[Parameter]):
    if cfg.optimizer == "adam":
        return Adam(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return SGD(
            params, lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(cfg.optimizer)


def create_optimizer_from_cfg(cfg: OptimizerConfig):
    return _.partial(create_optimizer, cfg)
