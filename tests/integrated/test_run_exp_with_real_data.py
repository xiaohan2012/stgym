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
from stgym.data_loader.ds_info import get_all_ds_names, get_info
from stgym.rct.run import run_exp
from stgym.utils import rm_dir_if_exists

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"

# Filter datasets by task type from ds_info.py
GRAPH_CLASSIFICATION_DATASETS = [
    ds_name
    for ds_name in get_all_ds_names()
    if get_info(ds_name)["task_type"] == "graph-classification"
]

NODE_CLASSIFICATION_DATASETS = [
    ds_name
    for ds_name in get_all_ds_names()
    if get_info(ds_name)["task_type"] == "node-classification"
]


class TestRunExpWithRealData:
    """Integration tests for run_exp function with real datasets"""

    n_clusters = 4

    @property
    def base_train_config(self):
        """Base training configuration with minimal epochs for testing"""
        return TrainConfig(
            optim=OptimizerConfig(),
            lr_schedule=LRScheduleConfig(type=None),
            max_epoch=1,  # Minimal for fast testing
        )

    @property
    def regular_dl_config(self):
        """Regular data loader configuration (non k-fold)"""
        cfg = DataLoaderConfig(batch_size=4)  # Small batch for testing
        cfg.split = DataLoaderConfig.DataSplitConfig(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        return cfg

    @property
    def kfold_dl_config(self):
        """K-fold data loader configuration"""
        cfg = DataLoaderConfig(batch_size=4)  # Small batch for testing
        cfg.split = DataLoaderConfig.KFoldSplitConfig(num_folds=3, split_index=0)
        return cfg

    @property
    def mlflow_config(self):
        """MLflow configuration with tracking disabled for testing"""
        return MLFlowConfig(track=False)

    @property
    def graph_classifier_model_config(self):
        """Base graph classifier model configuration"""
        return GraphClassifierModelConfig(
            mp_layers=[
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=self.n_clusters),
                )
            ],
            global_pooling="mean",
            post_mp_layer=PostMPConfig(dims=[16, 8]),
        )

    @property
    def node_classifier_model_config(self):
        """Base node classifier model configuration"""
        return NodeClassifierModelConfig(
            mp_layers=[
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=self.n_clusters),
                )
            ],
            post_mp_layer=PostMPConfig(dims=[16, 8]),
        )

    def _create_graph_clf_experiment_config(
        self, dataset_name, num_classes, use_kfold=False
    ):
        """Helper to create graph classification experiment config"""
        task_config = TaskConfig(
            dataset_name=dataset_name,
            type="graph-classification",
            num_classes=num_classes,
        )

        train_config = self.base_train_config.model_copy()
        train_config.early_stopping = TrainConfig.EarlyStoppingConfig(
            metric="val_roc_auc", mode="max"
        )

        dl_config = self.kfold_dl_config if use_kfold else self.regular_dl_config

        return ExperimentConfig(
            task=task_config,
            data_loader=dl_config,
            model=self.graph_classifier_model_config,
            train=train_config,
        )

    def _create_node_clf_experiment_config(
        self, dataset_name, num_classes, use_kfold=False
    ):
        """Helper to create node classification experiment config"""
        task_config = TaskConfig(
            dataset_name=dataset_name,
            type="node-classification",
            num_classes=num_classes,
        )

        train_config = self.base_train_config.model_copy()
        train_config.early_stopping = TrainConfig.EarlyStoppingConfig(
            metric="val_accuracy", mode="max"
        )

        dl_config = self.kfold_dl_config if use_kfold else self.regular_dl_config

        return ExperimentConfig(
            task=task_config,
            data_loader=dl_config,
            model=self.node_classifier_model_config,
            train=train_config,
        )

    @pytest.mark.parametrize("dataset_name", GRAPH_CLASSIFICATION_DATASETS)
    def test_graph_classification_datasets_regular_split(self, dataset_name):
        """Test graph classification datasets with regular train/val/test split"""
        num_classes = get_info(dataset_name)["num_classes"]
        exp_cfg = self._create_graph_clf_experiment_config(
            dataset_name, num_classes, use_kfold=False
        )
        result = run_exp(exp_cfg, self.mlflow_config)
        assert result is True

    @pytest.mark.parametrize("dataset_name", GRAPH_CLASSIFICATION_DATASETS)
    def test_graph_classification_datasets_kfold_split(self, dataset_name):
        """Test graph classification datasets with k-fold cross validation"""
        num_classes = get_info(dataset_name)["num_classes"]
        exp_cfg = self._create_graph_clf_experiment_config(
            dataset_name, num_classes, use_kfold=True
        )
        result = run_exp(exp_cfg, self.mlflow_config)
        assert result is True

    @pytest.mark.parametrize("dataset_name", NODE_CLASSIFICATION_DATASETS)
    def test_node_classification_datasets_regular_split(self, dataset_name):
        """Test node classification datasets with regular train/val/test split"""
        num_classes = get_info(dataset_name)["num_classes"]
        exp_cfg = self._create_node_clf_experiment_config(
            dataset_name, num_classes, use_kfold=False
        )
        result = run_exp(exp_cfg, self.mlflow_config)
        assert result is True

    @pytest.mark.parametrize("dataset_name", NODE_CLASSIFICATION_DATASETS)
    def test_node_classification_datasets_kfold_split(self, dataset_name):
        """Test node classification datasets with k-fold cross validation"""
        num_classes = get_info(dataset_name)["num_classes"]
        exp_cfg = self._create_node_clf_experiment_config(
            dataset_name, num_classes, use_kfold=True
        )
        result = run_exp(exp_cfg, self.mlflow_config)
        assert result is True

    def teardown_method(self):
        """Clean up test data after each test"""
        # Clean up any processed data that might have been created during tests
        test_datasets = GRAPH_CLASSIFICATION_DATASETS + NODE_CLASSIFICATION_DATASETS
        for dataset_name in test_datasets:
            rm_dir_if_exists(f"data/{dataset_name}/processed")
