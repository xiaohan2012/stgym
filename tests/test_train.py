import pytest

from stgym.config_schema import (
    ClusteringModelConfig,
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
def clustering_model_cfg():
    return ClusteringModelConfig(
        # message passing layers
        mp_layers=[
            MessagePassingConfig(
                layer_type="gcnconv",
                pooling=PoolingConfig(type="dmon", n_clusters=8),
            ),
        ],
    )


@pytest.fixture
def graph_clf_train_cfg():
    return TrainConfig(
        optim=OptimizerConfig(),
        lr_schedule=LRScheduleConfig(type=None),
        max_epoch=10,
        early_stopping={"metric": "val_pr_auc", "mode": "min"},
    )


@pytest.fixture
def clustering_train_cfg():
    return TrainConfig(
        optim=OptimizerConfig(),
        lr_schedule=LRScheduleConfig(type=None),
        max_epoch=10,
        early_stopping={"metric": "val_nmi", "mode": "min"},
    )


@pytest.fixture
def dl_cfg():
    return DataLoaderConfig(batch_size=8)


@pytest.fixture
def graph_clf_task_cfg():
    return TaskConfig(dataset_name="brca-test", type="graph-classification")


@pytest.fixture
def clustering_task_cfg():
    return TaskConfig(dataset_name="human-crc-test", type="node-clustering")


@pytest.fixture
def mlflow_cfg():
    return MLFlowConfig(track=False)


def test_train_on_graph_clf_task(
    graph_clf_task_cfg, dl_cfg, graph_clf_model_cfg, graph_clf_train_cfg, mlflow_cfg
):
    """for graph classification task"""

    data_module = STDataModule(graph_clf_task_cfg, dl_cfg)
    model_module = STGymModule(
        dim_in=data_module.num_features,
        model_cfg=graph_clf_model_cfg,
        train_cfg=graph_clf_train_cfg,
        task_cfg=graph_clf_task_cfg,
        dim_out=1,  # 1 for binary classification
    )

    train(model_module, data_module, graph_clf_train_cfg, mlflow_cfg)


def test_train_on_clustering_task(
    clustering_task_cfg, dl_cfg, clustering_model_cfg, clustering_train_cfg, mlflow_cfg
):
    """for node clustering task"""
    data_module = STDataModule(clustering_task_cfg, dl_cfg)
    model_module = STGymModule(
        dim_in=data_module.num_features,
        model_cfg=clustering_model_cfg,
        train_cfg=clustering_train_cfg,
        task_cfg=clustering_task_cfg,
    )

    train(model_module, data_module, clustering_train_cfg, mlflow_cfg)
