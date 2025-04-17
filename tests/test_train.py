import pytest
from stgym.train import train
from stgym.config_schema import (
    MessagePassingConfig,
    ModelConfig,
    PoolingConfig,
    PostMPConfig,
    TrainConfig,
    OptimizerConfig,
    LRScheduleConfig,
    DataLoaderConfig,
)
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule


@pytest.fixture
def model_cfg():
    return ModelConfig(
        # message passing layers
        mp_layers=[
            MessagePassingConfig(
                layer_type="gcnconv",
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


@pytest.fixture
def train_cfg():
    return TrainConfig(
        optim=OptimizerConfig(), lr_schedule=LRScheduleConfig(type=None), max_epoch=10
    )


@pytest.fixture
def data_cfg():
    return DataLoaderConfig(dataset_name="brca-test", batch_size=8)


def test_train(data_cfg, model_cfg, train_cfg):
    data_module = STDataModule(data_cfg)
    model_module = STGymModule(
        dim_in=data_module.num_features,
        dim_out=1,  # 1 for binary classification
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    train(model_module, data_module, train_cfg)
