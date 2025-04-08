from typing import Iterator

import pydash as _
from torch.nn import Parameter
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR

from stgym.config_schema import OptimizerConfig, LRScheduleConfig


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


def create_scheduler(optimizer: Optimizer, cfg: LRScheduleConfig):
    if cfg.type is None:
        return StepLR(optimizer, step_size=cfg.max_epoch + 1)
    else:
        raise ValueError(f"cfg: {cfg}")
