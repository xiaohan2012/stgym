#!/usr/bin/env python

from stgym.config_schema import (
    DataLoaderConfig,
    LRScheduleConfig,
    MessagePassingConfig,
    ModelConfig,
    OptimizerConfig,
    PoolingConfig,
    PostMPConfig,
    TrainConfig,
)
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule
from stgym.train import train

model_cfg = ModelConfig(
    # message passing layers
    mp_layers=[
        MessagePassingConfig(
            layer_type="gcnconv",
            hidden_dim=64,
            pooling=PoolingConfig(type="dmon", n_clusters=8),
        ),
        # # training works under one mp+pooling layer only, not more than that
        # MessagePassingConfig(
        #     layer_type="gcnconv",
        #     pooling=PoolingConfig(type="dmon", n_clusters=4),
        # ),
    ],
    global_pooling="mean",
    post_mp_layer=PostMPConfig(dims=[16, 8]),
)


train_cfg = TrainConfig(
    optim=OptimizerConfig(base_lr=0.1),
    lr_schedule=LRScheduleConfig(type=None),
    max_epoch=10,
)


data_cfg = DataLoaderConfig(dataset_name="brca", batch_size=8)


data_module = STDataModule(data_cfg)
model_module = STGymModule(
    dim_in=data_module.num_features,
    dim_out=1,  # 1 for binary classification
    model_cfg=model_cfg,
    train_cfg=train_cfg,
)
print(model_module.model)


train(model_module, data_module, train_cfg, trainer_config={"log_every_n_steps": 10})
