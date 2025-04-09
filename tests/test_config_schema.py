from stgym.config_schema import LayerConfig, PostMPConfig, TrainConfig


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
    cfg =TrainConfig(max_epoch=1000)
    assert cfg.lr_schedule.max_epoch == 1000
