import pytest

from stgym.config_schema import (
    LayerConfig,
    MemoryConfig,
    MessagePassingConfig,
    PoolingConfig,
    PostMPConfig,
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
        layer_config = LayerConfig(layer_type=layer_type)

        mem_config = MemoryConfig()
        batch = self.load_batch()
        layer = GeneralLayer(
            self.num_features,
            self.num_classes,
            layer_config,
            mem_config,
        ).to(self.device)
        assert isinstance(layer.layer, expected_layer_class)

        output = layer(batch)
        assert output.x.shape == (
            self.num_nodes_per_graph * self.batch_size,
            self.num_classes,
        )

    def test_with_pooling(self):
        layer_type = "sageconv"
        n_clusters = 5
        pooling_config = PoolingConfig(
            type="dmon",
            n_clusters=n_clusters,
        )
        inner_dim = 64
        layer_config = MessagePassingConfig(
            layer_type=layer_type, pooling=pooling_config
        )

        mem_config = MemoryConfig()
        batch = self.load_batch()
        layer = GeneralLayer(self.num_features, inner_dim, layer_config, mem_config)

        output_batch = layer(batch)
        assert output_batch.adj_t.shape == (
            self.batch_size * n_clusters,
            self.batch_size * n_clusters,
        )
        assert output_batch.x.shape == (self.batch_size * n_clusters, inner_dim)


class TestGeneralMultiLayer(BatchLoaderMixin):
    @pytest.mark.parametrize("dim_inner", [[128, 64]])
    def test_simple(self, dim_inner):
        n_layers = len(dim_inner)
        layer_configs = [
            MessagePassingConfig(layer_type="gcnconv", dim_inner=dim)
            for dim in dim_inner
        ]
        mem_config = MemoryConfig()

        model = GeneralMultiLayer(
            self.num_features,
            layer_configs,
            mem_config,
        )

        layers = list(model.children())
        assert len(layers) == n_layers

        batch = self.load_batch()

        output = model(batch)
        assert output.x.shape == (
            self.num_nodes_per_graph * self.batch_size,
            dim_inner[-1],
        )

    def test_with_pooling(self):
        # dim_inner = [128, 64]
        final_n_clusters = 10
        final_dim_inner = 64
        # n_layers = len(dim_inner)
        layer_configs = [
            MessagePassingConfig(
                layer_type="gcnconv",
                dim_inner=128,
                pooling=PoolingConfig(
                    type="dmon",
                    n_clusters=20,
                ),
            ),
            MessagePassingConfig(
                layer_type="gcnconv",
                dim_inner=final_dim_inner,
                pooling=PoolingConfig(
                    type="dmon",
                    n_clusters=final_n_clusters,
                ),
            ),
        ]
        n_layers = len(layer_configs)
        mem_config = MemoryConfig()
        model = GeneralMultiLayer(
            self.num_features,
            layer_configs,
            mem_config,
        )

        layers = list(model.children())
        assert len(layers) == n_layers

        batch = self.load_batch()

        output = model(batch)
        assert output.x.shape == (final_n_clusters * self.batch_size, final_dim_inner)
        assert output.s.shape == (self.batch_size * 20, final_n_clusters)
        assert output.adj_t.shape == (
            self.batch_size * final_n_clusters,
            self.batch_size * final_n_clusters,
        )

        # the loss should be accumulated in a list, each element corresponding to the one pooling layer
        assert isinstance(output.loss, list)
        assert len(output.loss) == 2


class TestMLP(BatchLoaderMixin):
    def check(self, model):
        layers = list(model.model.children())[0].children()
        for layer in layers:
            assert isinstance(layer.layer, Linear)

        batch = self.load_batch()
        output = model(batch)

        assert output.x.shape == (
            self.num_nodes_per_graph * self.batch_size,
            self.num_classes,
        )

    def test_layers_are_linear(self):
        layer_configs = [
            LayerConfig(layer_type="linear", dim_inner=64),
            LayerConfig(layer_type="linear", dim_inner=self.num_classes),
        ]
        mem_config = MemoryConfig()

        model = MLP(self.num_features, layer_configs, mem_config)
        self.check(model)

    def test_creation_from_post_mp_config(self):
        postmp_cfg = PostMPConfig(dims=[20, self.num_classes])

        mem_config = MemoryConfig()

        model = MLP(self.num_features, postmp_cfg, mem_config)
        self.check(model)
