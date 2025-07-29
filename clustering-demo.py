from stgym.config_schema import (
    ClusteringModelConfig,
    DataLoaderConfig,
    LRScheduleConfig,
    MessagePassingConfig,
    MLFlowConfig,
    OptimizerConfig,
    PoolingConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule
from stgym.train import train

model_cfg = ClusteringModelConfig(
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
)


task_cfg = TaskConfig(
    dataset_name="human-crc",
    type="node-clustering",
)
train_cfg = TrainConfig(
    optim=OptimizerConfig(),
    lr_schedule=LRScheduleConfig(type=None),
    max_epoch=2,
    early_stopping={"metric": "val_nmi", "mode": "max"},
)


data_cfg = DataLoaderConfig(batch_size=8)


data_module = STDataModule(task_cfg, data_cfg)
model_module = STGymModule(
    dim_in=data_module.num_features,
    task_cfg=task_cfg,
    model_cfg=model_cfg,
    train_cfg=train_cfg,
)
print(model_module.model)


mlflow_cfg = MLFlowConfig(
    track=True, tracking_uri="http://127.0.0.1:5000", experiment_name="clustering-demo"
)


train(
    model_module,
    data_module,
    train_cfg,
    mlflow_cfg,
    tl_train_config={"log_every_n_steps": 10},
)
