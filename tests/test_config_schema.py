from pytorch_lightning.loggers import MLFlowLogger

from stgym.config_schema import LayerConfig, MLFlowConfig, PostMPConfig, TrainConfig


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
