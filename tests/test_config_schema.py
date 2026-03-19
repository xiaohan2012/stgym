import pytest
from pydantic import ValidationError
from pytorch_lightning.loggers import MLFlowLogger

from stgym.config_schema import (
    _PYG_CONV_CLASSES,
    DataLoaderConfig,
    ExperimentConfig,
    GraphClassifierModelConfig,
    LayerConfig,
    MessagePassingConfig,
    MLFlowConfig,
    PoolingConfig,
    PostMPConfig,
    TaskConfig,
    TrainConfig,
    dataset_eval_mode,
    supports_edge_weight,
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


class TestEarlyStoppingMetric:
    """Test that early stopping metric remains generic across all split types."""

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

    def test_kfold_split_uses_generic_metric(self):
        # Test with kfold split - metric should remain generic
        exp_config = self.create_exp_config(
            DataLoaderConfig(split=dict(num_folds=5, split_index=2))
        )

        # Check that early stopping metric stays generic (no split prefix)
        assert exp_config.train.early_stopping.metric == "val_loss"

        # Validate again with different split_index - metric should still be generic
        exp_config.data_loader.split.split_index = 3
        exp_config = exp_config.validate()
        assert exp_config.train.early_stopping.metric == "val_loss"

    def test_regular_split_uses_generic_metric(self):
        # Test with regular split - metric should remain generic
        exp_config_regular = self.create_exp_config(
            DataLoaderConfig(
                split=dict(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
            )
        )
        # Check that early stopping metric is generic
        assert exp_config_regular.train.early_stopping.metric == "val_loss"


class TestEdgeWeightPoolingValidation:
    """Validate that unweighted MP operators are rejected with multi-layer pooling."""

    def _make_mp_layer(self, layer_type, with_pooling=False):
        pooling = PoolingConfig(type="dmon", n_clusters=3) if with_pooling else None
        return MessagePassingConfig(layer_type=layer_type, pooling=pooling)

    @pytest.mark.parametrize("layer_type", ["sageconv", "ginconv", "gcnconv"])
    def test_single_pooling_layer_accepts_any_operator(self, layer_type):
        """A single pooling layer is fine regardless of operator."""
        cfg = GraphClassifierModelConfig(
            mp_layers=[
                self._make_mp_layer(layer_type, with_pooling=True),
                self._make_mp_layer(layer_type, with_pooling=False),
            ],
            post_mp_layer=PostMPConfig(dims=[10]),
        )
        assert len([mp for mp in cfg.mp_layers if mp.has_pooling]) == 1

    @pytest.mark.parametrize(
        "layer_type",
        [k for k in _PYG_CONV_CLASSES if supports_edge_weight(k)],
    )
    def test_multi_pooling_with_weighted_operator_is_valid(self, layer_type):
        """Operators that support edge_weight are fine with multi-pooling."""
        cfg = GraphClassifierModelConfig(
            mp_layers=[
                self._make_mp_layer(layer_type, with_pooling=True),
                self._make_mp_layer(layer_type, with_pooling=True),
            ],
            post_mp_layer=PostMPConfig(dims=[10]),
        )
        assert len([mp for mp in cfg.mp_layers if mp.has_pooling]) == 2

    @pytest.mark.parametrize("layer_type", ["sageconv", "ginconv"])
    def test_multi_pooling_with_unweighted_operator_raises(self, layer_type):
        """SAGEConv/GINConv don't support edge_weight — reject with multi-pooling."""
        with pytest.raises(ValidationError, match="edge_weight"):
            GraphClassifierModelConfig(
                mp_layers=[
                    self._make_mp_layer(layer_type, with_pooling=True),
                    self._make_mp_layer(layer_type, with_pooling=True),
                ],
                post_mp_layer=PostMPConfig(dims=[10]),
            )


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
