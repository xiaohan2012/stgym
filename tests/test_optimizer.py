import torch.nn as nn
from torch.optim import Adam

from stgym.config_schema import OptimizerConfig
from stgym.optimizer import create_optimizer_from_cfg


def test_load_optimizer():
    cfg = OptimizerConfig(optimizer="adam", lr=0.001, weight_decay=0.0001)
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    create_optimizer = create_optimizer_from_cfg(cfg)
    optimizer = create_optimizer(model.parameters())
    assert isinstance(optimizer, Adam)
