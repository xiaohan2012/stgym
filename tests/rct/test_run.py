from unittest.mock import Mock, patch

import pytest
import torch

from stgym.config_schema import (
    ClusteringModelConfig,
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
from stgym.rct.run import run_exp
from stgym.utils import rm_dir_if_exists

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


@pytest.fixture
def base_task_configs():
    """Base task configurations for different task types"""
    return {
        "graph_clf": TaskConfig(
            dataset_name="brca-test", type="graph-classification", num_classes=2
        ),
        "graph_clf_multiclass": TaskConfig(
            dataset_name="brca-test", type="graph-classification", num_classes=5
        ),
        "node_clf": TaskConfig(
            dataset_name="human-crc-test", type="node-classification", num_classes=10
        ),
        "clustering": TaskConfig(dataset_name="human-crc-test", type="node-clustering"),
    }


@pytest.fixture
def base_model_configs():
    """Base model configurations for different model types"""
    return {
        "graph_clf": GraphClassifierModelConfig(
            mp_layers=[
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=8),
                )
            ],
            global_pooling="mean",
            post_mp_layer=PostMPConfig(dims=[16, 8]),
        ),
        "node_clf": NodeClassifierModelConfig(
            mp_layers=[
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=8),
                )
            ],
            post_mp_layer=PostMPConfig(dims=[16, 8]),
        ),
        "clustering": ClusteringModelConfig(
            mp_layers=[
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=8),
                )
            ]
        ),
    }


@pytest.fixture
def base_train_configs():
    """Base training configurations for different task types"""
    return {
        "graph_clf": TrainConfig(
            optim=OptimizerConfig(),
            lr_schedule=LRScheduleConfig(type=None),
            max_epoch=2,  # Reduced for testing
            early_stopping={"metric": "val_roc_auc", "mode": "max"},
        ),
        "node_clf": TrainConfig(
            optim=OptimizerConfig(),
            lr_schedule=LRScheduleConfig(type=None),
            max_epoch=2,  # Reduced for testing
            early_stopping={"metric": "val_accuracy", "mode": "max"},
        ),
        "clustering": TrainConfig(
            optim=OptimizerConfig(),
            lr_schedule=LRScheduleConfig(type=None),
            max_epoch=2,  # Reduced for testing
            early_stopping={"metric": "val_nmi", "mode": "max"},
        ),
    }


@pytest.fixture
def regular_dl_config():
    """Regular data loader configuration (non k-fold)"""
    cfg = DataLoaderConfig(batch_size=4)  # Reduced for testing
    cfg.split = DataLoaderConfig.DataSplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    return cfg


@pytest.fixture
def kfold_dl_config():
    """K-fold data loader configuration"""
    cfg = DataLoaderConfig(batch_size=4)  # Reduced for testing
    cfg.split = DataLoaderConfig.KFoldSplitConfig(num_folds=3, split_index=0)
    return cfg


@pytest.fixture
def mlflow_config():
    """MLflow configuration with tracking disabled for testing"""
    return MLFlowConfig(track=False)


class TestRunExp:
    """Test class for the run_exp function covering both k-fold and regular splits"""

    @pytest.fixture(autouse=True)
    def setup_configs(
        self,
        base_task_configs,
        base_model_configs,
        base_train_configs,
        regular_dl_config,
        kfold_dl_config,
    ):
        """Auto-use fixture to make configs available as class attributes"""
        self.base_task_configs = base_task_configs
        self.base_model_configs = base_model_configs
        self.base_train_configs = base_train_configs
        self.regular_dl_config = regular_dl_config
        self.kfold_dl_config = kfold_dl_config

    @pytest.fixture(autouse=True)
    def teardown(self):
        """Auto-use fixture to clean up test data after each test"""
        yield  # Run the test
        # Clean up test data
        rm_dir_if_exists("tests/data/brca-test/processed")
        rm_dir_if_exists("tests/data/human-crc-test/processed")

    def _create_experiment_config(
        self, task_key, data_loader_type="regular", model_key=None, group_id=None
    ):
        """Helper method to create ExperimentConfig with common pattern"""
        model_key = model_key or task_key

        # Resolve data loader based on type
        if data_loader_type == "regular":
            data_loader = self.regular_dl_config
        elif data_loader_type == "kfold":
            data_loader = self.kfold_dl_config
        else:
            raise ValueError(f"Unknown data_loader_type: {data_loader_type}")

        config_args = {
            "task": self.base_task_configs[task_key],
            "data_loader": data_loader,
            "model": self.base_model_configs[model_key],
            "train": self.base_train_configs[model_key],
        }

        if group_id is not None:
            config_args["group_id"] = group_id

        return ExperimentConfig(**config_args)

    @pytest.mark.parametrize(
        "split_type,expected_calls",
        [
            ("regular", 1),
            ("kfold", 3),  # num_folds from kfold_dl_config
        ],
    )
    def test_kfold_vs_regular_splits(
        self,
        split_type,
        expected_calls,
        mlflow_config,
    ):
        """Test run_exp with different split types for graph classification"""
        exp_cfg = self._create_experiment_config(
            "graph_clf", data_loader_type=split_type
        )

        with patch("stgym.rct.run.train") as mock_train:
            mock_train.return_value = None
            result = run_exp(exp_cfg, mlflow_config)

        assert result is True
        assert mock_train.call_count == expected_calls

    @pytest.mark.parametrize(
        "task_key,expected_dim_out",
        [
            ("graph_clf", 1),  # binary classification: num_classes=2 -> dim_out=1
            ("graph_clf_multiclass", 5),  # multiclass: num_classes=5 -> dim_out=5
        ],
    )
    def test_binary_vs_multiclass_output_dim(
        self,
        task_key,
        expected_dim_out,
        mlflow_config,
    ):
        """Test that binary classification uses dim_out=1 while multiclass uses num_classes"""
        exp_cfg = self._create_experiment_config(task_key, model_key="graph_clf")

        with patch("stgym.rct.run.STGymModule") as mock_module_class, patch(
            "stgym.rct.run.train"
        ):
            mock_module_class.return_value = Mock()
            run_exp(exp_cfg, mlflow_config)

            # Check that dim_out matches expected value
            call_args = mock_module_class.call_args
            assert call_args[1]["dim_out"] == expected_dim_out

    def test_node_classification(
        self,
        mlflow_config,
    ):
        """Test run_exp with node classification task"""
        exp_cfg = self._create_experiment_config("node_clf")

        with patch("stgym.rct.run.train") as mock_train:
            mock_train.return_value = None
            result = run_exp(exp_cfg, mlflow_config)

        assert result is True
        mock_train.assert_called_once()

    def test_clustering_task(
        self,
        mlflow_config,
    ):
        """Test run_exp with clustering task (dim_out should be None)"""
        exp_cfg = self._create_experiment_config("clustering")

        with patch("stgym.rct.run.STGymModule") as mock_module_class, patch(
            "stgym.rct.run.train"
        ):
            mock_module_class.return_value = Mock()
            result = run_exp(exp_cfg, mlflow_config)

            # Check that dim_out=None for clustering
            call_args = mock_module_class.call_args
            assert call_args[1]["dim_out"] is None

        assert result is True

    @pytest.mark.parametrize(
        "mock_target,exception_msg",
        [
            ("stgym.rct.run.train", "Training failed"),
            ("stgym.rct.run.STDataModule", "Data loading failed"),
        ],
    )
    @patch("stgym.rct.run.logz_logger")
    def test_error_handling(
        self,
        mock_logger,
        mlflow_config,
        mock_target,
        exception_msg,
    ):
        """Test error handling during training and data loading"""
        exp_cfg = self._create_experiment_config("graph_clf")

        # Mock the specified target to raise an exception
        with patch(mock_target) as mock_component:
            mock_component.side_effect = RuntimeError(exception_msg)

            result = run_exp(exp_cfg, mlflow_config)

            # Should still return True even with error
            assert result is True
            # Error should be logged
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert exception_msg in error_call
