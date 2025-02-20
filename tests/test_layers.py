import pytest

from stgym.config_schema import LayerConfig, MemoryConfig
from stgym.layers import (
    MLP,
    GCNConv,
    GeneralLayer,
    GeneralMultiLayer,
    GINConv,
    Linear,
    SAGEConv,
)

from .utils import create_data_batch


class BatchLoaderMixin:
    num_nodes = 100
    num_features = 128
    num_classes = 10
    batch_size = 2

    def load_batch(self):
        return create_data_batch(
            num_nodes=self.num_nodes,
            num_features=self.num_features,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
        )


class TestGeneralLayer(BatchLoaderMixin):
    @pytest.mark.parametrize(
        "layer_type, expected_layer_class",
        [
            ("gcnconv", GCNConv),
            ("sageconv", SAGEConv),
            ("ginconv", GINConv),
            ("linear", Linear),
        ],
    )
    def test(self, layer_type, expected_layer_class):
        layer_config = LayerConfig()
        mem_config = MemoryConfig()
        batch = self.load_batch()
        layer = GeneralLayer(
            layer_type, self.num_features, self.num_classes, layer_config, mem_config
        )
        assert isinstance(layer.layer, expected_layer_class)

        output = layer(batch)
        assert output.x.shape == (self.num_nodes * self.batch_size, 10)


class TestGeneralMultiLayer(BatchLoaderMixin):
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_simple(self, num_layers):
        layer_config = LayerConfig(n_layers=num_layers, dim_inner=64)
        mem_config = MemoryConfig()

        model = GeneralMultiLayer(
            "gcnconv", self.num_features, self.num_classes, layer_config, mem_config
        )

        layers = list(model.children())
        assert len(layers) == num_layers

        batch = self.load_batch()

        output = model(batch)

        assert output.x.shape == (self.num_nodes * self.batch_size, self.num_classes)


class TestMLP(BatchLoaderMixin):
    def test_layers_are_linear(self):
        layer_config = LayerConfig(n_layers=3, dim_inner=64)
        mem_config = MemoryConfig()

        model = MLP(self.num_features, self.num_classes, layer_config, mem_config)

        layers = list(model.model.children())[0].children()
        for layer in layers:
            assert isinstance(layer.layer, Linear)

        batch = self.load_batch()
        output = model(batch)

        assert output.x.shape == (self.num_nodes * self.batch_size, self.num_classes)
