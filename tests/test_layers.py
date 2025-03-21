import pytest

from stgym.config_schema import (
    LayerConfig,
    MemoryConfig,
    MessagePassingConfig,
    MultiLayerConfig,
    PoolingConfig,
)
from stgym.layers import (
    MLP,
    GCNConv,
    GeneralLayer,
    GeneralMultiLayer,
    GINConv,
    Linear,
    SAGEConv,
)

from .utils import BatchLoaderMixin


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
    def test_without_pooling(self, layer_type, expected_layer_class):
        layer_config = LayerConfig()

        mem_config = MemoryConfig()
        batch = self.load_batch()
        layer = GeneralLayer(
            layer_type, self.num_features, self.num_classes, layer_config, mem_config
        )
        assert isinstance(layer.layer, expected_layer_class)

        output = layer(batch)
        assert output.x.shape == (self.num_nodes * self.batch_size, self.num_classes)

    def test_with_pooling(self):
        layer_type = "sageconv"
        n_clusters = 5
        pooling_config = PoolingConfig(
            type="dmon",
            n_clusters=n_clusters,
        )
        inner_dim = 64
        layer_config = MessagePassingConfig(pooling=pooling_config)

        mem_config = MemoryConfig()
        batch = self.load_batch()
        layer = GeneralLayer(
            layer_type, self.num_features, inner_dim, layer_config, mem_config
        )

        output_batch = layer(batch)
        assert output_batch.adj.shape == (self.batch_size, n_clusters, n_clusters)
        assert output_batch.x.shape == (self.batch_size, n_clusters, inner_dim)


# @pytest.mark.skip(reason="to fix!")
class TestGeneralMultiLayer(BatchLoaderMixin):
    @pytest.mark.parametrize("dim_inner", [[128, 64]])
    def test_simple(self, dim_inner):
        n_layers = len(dim_inner)
        multi_layer_config = MultiLayerConfig(
            layers=[LayerConfig(dim_inner=dim) for dim in dim_inner]
        )
        mem_config = MemoryConfig()

        model = GeneralMultiLayer(
            "gcnconv",
            self.num_features,
            multi_layer_config,
            mem_config,
        )

        layers = list(model.children())
        assert len(layers) == n_layers

        batch = self.load_batch()

        output = model(batch)

        assert output.x.shape == (self.num_nodes * self.batch_size,)


@pytest.mark.skip(reason="to fix!")
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
