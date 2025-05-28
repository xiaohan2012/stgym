#!/usr/bin/env python

from stgym.config_schema import (
    DataLoaderConfig,
    GraphClassifierModelConfig,
    LRScheduleConfig,
    MessagePassingConfig,
    MLFlowConfig,
    OptimizerConfig,
    PoolingConfig,
    PostMPConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule
from stgym.train import train

model_cfg = GraphClassifierModelConfig(
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
    max_epoch=2,
)


task_cfg = TaskConfig(dataset_name="brca", type="graph-classification")
dl_cfg = DataLoaderConfig(batch_size=8)
mlflow_cfg = MLFlowConfig(
    track=True, tracking_uri="http://127.0.0.1:8080", experiment_name="train-demo"
)

data_module = STDataModule(task_cfg, dl_cfg)
model_module = STGymModule(
    dim_in=data_module.num_features,
    dim_out=1,  # 1 for binary classification
    model_cfg=model_cfg,
    train_cfg=train_cfg,
    task_cfg=task_cfg,
)
print(model_module)


train(
    model_module,
    data_module,
    train_cfg,
    mlflow_cfg,
    tl_train_config={"log_every_n_steps": 10},
)
