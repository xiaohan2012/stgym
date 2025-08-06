from stgym.config_schema import (
    DataLoaderConfig,
    LRScheduleConfig,
    MessagePassingConfig,
    MLFlowConfig,
    NodeClassifierModelConfig,
    OptimizerConfig,
    PostMPConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.data_loader import STDataModule
from stgym.data_loader.ds_info import get_info
from stgym.tl_model import STGymModule
from stgym.train import train

model_cfg = NodeClassifierModelConfig(
    # message passing layers
    mp_layers=[
        MessagePassingConfig(layer_type="gcnconv", hidden_dim=64, pooling=None),
        MessagePassingConfig(layer_type="gcnconv", hidden_dim=32, pooing=None),
    ],
    post_mp_layer=PostMPConfig(dims=[16, 8]),
)


# ds_name = "mouse-spleen"
# ds_name = "human-intestine"
# ds_name = "human-lung"
ds_name = "breast-cancer"

task_cfg = TaskConfig(
    dataset_name=ds_name,
    type="node-classification",
    num_classes=get_info(ds_name)["num_classes"],
)
train_cfg = TrainConfig(
    optim=OptimizerConfig(),
    lr_schedule=LRScheduleConfig(type=None),
    max_epoch=10,
    early_stopping={"metric": "val_accuracy", "mode": "max"},
)


data_cfg = DataLoaderConfig(batch_size=8)


data_module = STDataModule(task_cfg, data_cfg)
model_module = STGymModule(
    dim_in=data_module.num_features,
    dim_out=task_cfg.num_classes,
    task_cfg=task_cfg,
    model_cfg=model_cfg,
    train_cfg=train_cfg,
)
print(model_module.model)


mlflow_cfg = MLFlowConfig(
    track=True, tracking_uri="http://127.0.0.1:5000", experiment_name="node-clf-demo"
)


train(
    model_module,
    data_module,
    train_cfg,
    mlflow_cfg,
    tl_train_config={"log_every_n_steps": 10},
)
