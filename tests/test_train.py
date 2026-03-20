import pytest
import torch

from stgym.config_schema import (
    DataLoaderConfig,
    ExperimentConfig,
    GraphClassifierModelConfig,
    LRScheduleConfig,
    MessagePassingConfig,
    MLFlowConfig,
    NodeClassifierModelConfig,
    OptimizerConfig,
    PoolingConfig,
    PostMPConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.data_loader import STDataModule, STKfoldDataModule
from stgym.tl_model import STGymModule
from stgym.train import train
from stgym.utils import rm_dir_if_exists

# device to store tensors on
TORCH_DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"

# devices passed to torch lightening Trainer
PL_TRAIN_DEVICES = (
    "auto" if TORCH_DEVICE == "cpu" else [0]
)  # to avoid DDP trainig in unittests on CUDA devices


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
def node_clf_model_cfg():
    return NodeClassifierModelConfig(
        mp_layers=[
            MessagePassingConfig(layer_type="gcnconv"),
        ],
        post_mp_layer=PostMPConfig(dims=[16, 8]),
    )


@pytest.fixture
def graph_clf_train_cfg():
    return TrainConfig(
        optim=OptimizerConfig(),
        lr_schedule=LRScheduleConfig(type=None),
        max_epoch=10,
        early_stopping={"metric": "val_roc_auc", "mode": "max"},
        devices=PL_TRAIN_DEVICES,
    )


@pytest.fixture
def node_clf_train_cfg():
    return TrainConfig(
        optim=OptimizerConfig(),
        lr_schedule=LRScheduleConfig(type=None),
        max_epoch=10,
        early_stopping={"metric": "val_accuracy", "mode": "max"},
        devices=PL_TRAIN_DEVICES,
    )


@pytest.fixture
def dl_cfg():
    cfg = DataLoaderConfig(batch_size=8)
    cfg.split = DataLoaderConfig.DataSplitConfig()
    return cfg


@pytest.fixture
def kfold_dl_cfg():
    cfg = DataLoaderConfig(batch_size=8)
    cfg.split = DataLoaderConfig.KFoldSplitConfig(num_folds=3, split_index=0)
    return cfg


@pytest.fixture
def graph_clf_task_cfg():
    return TaskConfig(
        dataset_name="brca-test", type="graph-classification", num_classes=2
    )


@pytest.fixture
def node_clf_task_cfg():
    return TaskConfig(
        dataset_name="human-crc-test", type="node-classification", num_classes=10
    )


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
        dl_cfg=dl_cfg,
        dim_out=1,  # 1 for binary classification
    ).to(TORCH_DEVICE)
    train(model_module, data_module, graph_clf_train_cfg, mlflow_cfg, logger=None)
    rm_dir_if_exists("tests/data/brca-test/processed")


def test_train_on_node_clf_task(
    node_clf_task_cfg, dl_cfg, node_clf_model_cfg, node_clf_train_cfg, mlflow_cfg
):
    """for node classification task"""

    data_module = STDataModule(node_clf_task_cfg, dl_cfg)
    model_module = STGymModule(
        dim_in=data_module.num_features,
        model_cfg=node_clf_model_cfg,
        train_cfg=node_clf_train_cfg,
        task_cfg=node_clf_task_cfg,
        dl_cfg=dl_cfg,
        dim_out=node_clf_task_cfg.num_classes,
    ).to(TORCH_DEVICE)

    train(model_module, data_module, node_clf_train_cfg, mlflow_cfg, logger=None)
    rm_dir_if_exists("tests/data/human-crc-test/processed")


class TestUsingKfoldData:
    def test_graph_clf_task_basic_functionality(
        self,
        graph_clf_task_cfg,
        kfold_dl_cfg,
        graph_clf_model_cfg,
        graph_clf_train_cfg,
        mlflow_cfg,
    ):
        """for graph classification task"""
        # trigger the model_validator logic
        # kind of a hack
        ExperimentConfig(
            task=graph_clf_task_cfg,
            data_loader=kfold_dl_cfg,
            model=graph_clf_model_cfg,
            train=graph_clf_train_cfg,
        )
        data_module = STKfoldDataModule(graph_clf_task_cfg, kfold_dl_cfg)
        model_module = STGymModule(
            dim_in=data_module.num_features,
            model_cfg=graph_clf_model_cfg,
            train_cfg=graph_clf_train_cfg,
            task_cfg=graph_clf_task_cfg,
            dl_cfg=kfold_dl_cfg,
            dim_out=1,  # 1 for binary classification
        ).to(TORCH_DEVICE)
        train(model_module, data_module, graph_clf_train_cfg, mlflow_cfg, logger=None)
        rm_dir_if_exists("tests/data/brca-test/processed")


def test_data_stays_on_cpu_and_lightning_transfers_to_device(
    graph_clf_task_cfg, dl_cfg, graph_clf_model_cfg, graph_clf_train_cfg, mlflow_cfg
):
    """Verify that dataset stays on CPU and Lightning transfers batches to the correct device."""
    data_module = STDataModule(graph_clf_task_cfg, dl_cfg)

    # 1. Verify dataset stays on CPU after construction
    for data in data_module.ds:
        assert data.x.device == torch.device("cpu"), (
            f"Dataset should stay on CPU, but found {data.x.device}"
        )
        break  # checking one sample is sufficient

    # 2. Verify batches arrive on the expected device during training
    #    by monkey-patching training_step to record the batch device
    observed_devices = []
    model_module = STGymModule(
        dim_in=data_module.num_features,
        model_cfg=graph_clf_model_cfg,
        train_cfg=graph_clf_train_cfg,
        task_cfg=graph_clf_task_cfg,
        dl_cfg=dl_cfg,
        dim_out=1,
    ).to(TORCH_DEVICE)

    original_training_step = model_module.training_step

    def patched_training_step(batch, *args, **kwargs):
        observed_devices.append(batch.x.device)
        return original_training_step(batch, *args, **kwargs)

    model_module.training_step = patched_training_step

    train(model_module, data_module, graph_clf_train_cfg, mlflow_cfg, logger=None)

    assert len(observed_devices) > 0, "training_step was never called"
    expected_type = torch.device(TORCH_DEVICE).type
    for dev in observed_devices:
        assert dev.type == expected_type, (
            f"Batch should be on {expected_type}, but found {dev}"
        )

    rm_dir_if_exists("tests/data/brca-test/processed")
