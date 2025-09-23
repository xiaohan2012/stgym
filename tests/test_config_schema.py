import pytest
from pydantic import ValidationError
from pytorch_lightning.loggers import MLFlowLogger

from stgym.config_schema import (
    DataLoaderConfig,
    ExperimentConfig,
    GraphClassifierModelConfig,
    LayerConfig,
    MessagePassingConfig,
    MLFlowConfig,
    PostMPConfig,
    TaskConfig,
    TrainConfig,
    dataset_eval_mode,
)


class TestPostMPConfig:
    def test(self):
        dropout, act = 0.5, "relu"
        dims = [20, 10]
        config = PostMPConfig(dims=dims, dropout=dropout, act=act)
        layer_configs = config.to_layer_configs()
        assert len(layer_configs) == len(dims)
        for layer_config in layer_configs:
            isinstance(layer_config, LayerConfig)
            assert layer_config.layer_type == "linear"
            assert layer_config.dropout == dropout
            assert layer_config.act == act


def test_train_config():
    cfg = TrainConfig(max_epoch=1000)
    assert cfg.lr_schedule.max_epoch == 1000


def test_mlflow_config_create_tl_logger():
    """Test MLFlowConfig.create_tl_logger method"""

    # Test when tracking is enabled (track=True)
    config_with_tracking = MLFlowConfig(track=True)

    logger = config_with_tracking.create_tl_logger()

    # Should return an MLFlowLogger instance
    assert logger is not None
    assert isinstance(logger, MLFlowLogger)

    # Test when tracking is disabled (track=False)
    config_without_tracking = MLFlowConfig(track=False)

    logger_disabled = config_without_tracking.create_tl_logger()

    # Should return None
    assert logger_disabled is None


class TestDataLoaderConfig:
    def test_use_kfold_split(self):
        cfg = DataLoaderConfig(split=dict(num_folds=10, split_index=0))
        assert cfg.use_kfold_split

        cfg = DataLoaderConfig(
            split=dict(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        )
        assert not cfg.use_kfold_split

    def test_validation(self):
        # Test valid split_index (should not raise)
        cfg = DataLoaderConfig(
            split=DataLoaderConfig.KFoldSplitConfig(num_folds=5, split_index=0)
        )
        assert cfg.split.split_index == 0

        cfg = DataLoaderConfig(
            split=DataLoaderConfig.KFoldSplitConfig(num_folds=5, split_index=4)
        )
        assert cfg.split.split_index == 4

        # Test invalid split_index (should raise ValidationError)
        with pytest.raises(
            ValidationError, match="split_index \\(5\\) must be in range \\[0, 5\\)"
        ):
            # this does not trigger validation logic
            # DataLoaderConfig(split=dict(num_folds=5, split_index=5))
            DataLoaderConfig(
                split=DataLoaderConfig.KFoldSplitConfig(num_folds=5, split_index=5)
            )

        with pytest.raises(
            ValidationError, match="split_index \\(10\\) must be in range \\[0, 3\\)"
        ):
            DataLoaderConfig(
                split=DataLoaderConfig.KFoldSplitConfig(num_folds=3, split_index=10)
            )


class TestearlyStoppingModificationLogic:
    """Test ExperimentConfig.modify_early_stopping_metric_when_kfold_split_is_used."""

    def create_exp_config(self, data_loader_cfg: DataLoaderConfig):
        return ExperimentConfig(
            task=TaskConfig(
                dataset_name="test", type="graph-classification", num_classes=2
            ),
            data_loader=data_loader_cfg,
            model=GraphClassifierModelConfig(
                mp_layers=[MessagePassingConfig(layer_type="gcnconv")],
                post_mp_layer=PostMPConfig(dims=[10]),
            ),
            train=TrainConfig(max_epoch=100),
        )

    def test_modification_fired(self):
        # Test with kfold split - metric should be modified
        exp_config = self.create_exp_config(
            DataLoaderConfig(split=dict(num_folds=5, split_index=2))
        )

        # Check that early stopping metric was modified to include split index
        assert exp_config.train.early_stopping.metric == "split_2_val_loss"

        # fire again
        exp_config.data_loader.split.split_index = 3
        exp_config = exp_config.validate()
        assert exp_config.train.early_stopping.metric == "split_3_val_loss"

    def test_modification_silent(self):
        # Test with regular split - metric should not be modified
        exp_config_regular = self.create_exp_config(
            DataLoaderConfig(
                split=dict(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
            )
        )
        # Check that early stopping metric was not modified
        assert exp_config_regular.train.early_stopping.metric == "val_loss"


class TestExperimentConfig:
    @pytest.mark.parametrize("ds_name", list(dataset_eval_mode.keys()))
    def test_override_eval_mode(self, ds_name: str):
        cfg = ExperimentConfig(
            task=TaskConfig(
                dataset_name=ds_name,
                type="node-classification",
                num_classes=2,
            ),
            data_loader=DataLoaderConfig(split=DataLoaderConfig.DataSplitConfig()),
            model=GraphClassifierModelConfig(
                mp_layers=[MessagePassingConfig(layer_type="gcnconv")],
                post_mp_layer=PostMPConfig(dims=[10]),
            ),
            train=TrainConfig(max_epoch=100),
        )
        assert cfg.data_loader.split == dataset_eval_mode[ds_name]
