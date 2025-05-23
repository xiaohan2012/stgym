import pytest

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


@pytest.fixture
def graph_clf_model_cfg():
    return GraphClassifierModelConfig(
        # message passing layers
        mp_layers=[
            MessagePassingConfig(
                layer_type="gcnconv",
                pooling=PoolingConfig(type="dmon", n_clusters=8),
                # pooling=PoolingConfig(type="mincut", n_clusters=8),
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
def node_cls_model_cfg():
    return GraphClassifierModelConfig(
        # message passing layers
        mp_layers=[
            MessagePassingConfig(
                layer_type="gcnconv",
                pooling=PoolingConfig(type="dmon", n_clusters=8),
                # pooling=PoolingConfig(type="mincut", n_clusters=8),
            ),
            # # training works under one mp+pooling layer only, not more than that
            # MessagePassingConfig(
            #     layer_type="gcnconv",
            #     pooling=PoolingConfig(type="dmon", n_clusters=4),
            # ),
        ],
        # global_pooling="mean",
        post_mp_layer=PostMPConfig(dims=[16, 8]),
    )


@pytest.fixture
def train_cfg():
    return TrainConfig(
        optim=OptimizerConfig(),
        lr_schedule=LRScheduleConfig(type=None),
        max_epoch=10,
        early_stopping={"metric": "val_pr_auc", "mode": "min"},
    )


@pytest.fixture
def dl_cfg():
    return DataLoaderConfig(batch_size=8)


@pytest.fixture
def graph_clf_task_cfg():
    return TaskConfig(dataset_name="brca-test", type="graph-classification")


@pytest.fixture
def node_cls_task_cfg():
    return TaskConfig(dataset_name="human-crc-test", type="node-clustering")


@pytest.fixture
def mlflow_cfg():
    return MLFlowConfig(track=False)


def test_train_on_graph_cfg_task(
    graph_clf_task_cfg, dl_cfg, graph_clf_model_cfg, train_cfg, mlflow_cfg
):
    """for graph classification task"""

    data_module = STDataModule(graph_clf_task_cfg, dl_cfg)
    model_module = STGymModule(
        dim_in=data_module.num_features,
        dim_out=1,  # 1 for binary classification
        model_cfg=graph_clf_model_cfg,
        train_cfg=train_cfg,
        task_cfg=graph_clf_task_cfg,
    )

    train(model_module, data_module, train_cfg, mlflow_cfg)


def test_train_on_node_cls_task(
    node_cls_task_cfg, dl_cfg, node_cls_model_cfg, train_cfg, mlflow_cfg
):
    """for node clustering task"""
    data_module = STDataModule(node_cls_task_cfg, dl_cfg)
    model_module = STGymModule(
        dim_in=data_module.num_features,
        dim_out=node_cls_model_cfg.mp_layers,  # TODO: what should be the output dim
        model_cfg=node_cls_model_cfg,
        train_cfg=train_cfg,
        task_cfg=node_cls_task_cfg,
    )

    train(model_module, data_module, train_cfg, mlflow_cfg)
