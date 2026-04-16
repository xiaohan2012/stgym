import pytest
from torch import Tensor
from torch_geometric.data import Data

from stgym.config_schema import (
    GraphClassifierModelConfig,
    MessagePassingConfig,
    NodeClassifierModelConfig,
    PoolingConfig,
    PostMPConfig,
)
from stgym.model import STGraphClassifier, STNodeClassifier

from .utils import BatchLoaderMixin


class TestSTGraphClassifier(BatchLoaderMixin):
    def test(self):
        cfg = GraphClassifierModelConfig(
            mp_layers=[
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=128),
                ),
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=32),
                ),
            ],
            global_pooling="mean",
            post_mp_layer=PostMPConfig(dims=[64, 32]),
        )
        batch = self.load_batch()
        model = STGraphClassifier(self.num_features, self.num_classes, cfg).to(
            self.device
        )
        batch, pred, other_loss = model(batch)
        assert isinstance(pred, Tensor)
        assert isinstance(batch, Data)
        assert pred.shape == (self.batch_size, self.num_classes)
        assert len(other_loss) == 2
        for loss in other_loss:
            assert isinstance(loss, dict)


@pytest.mark.parametrize(
    "model_cls,cfg_factory",
    [
        (
            STGraphClassifier,
            lambda: GraphClassifierModelConfig(
                mp_layers=[MessagePassingConfig(layer_type="gcnconv", pooling=None)],
                global_pooling="mean",
                post_mp_layer=PostMPConfig(dims=[32]),
            ),
        ),
        (
            STNodeClassifier,
            lambda: NodeClassifierModelConfig(
                mp_layers=[MessagePassingConfig(layer_type="gcnconv", pooling=None)],
                post_mp_layer=PostMPConfig(dims=[32]),
            ),
        ),
    ],
    ids=["graph_clf", "node_clf"],
)
class TestConfigImmutability(BatchLoaderMixin):
    """Verify that model construction does not mutate the shared PostMPConfig."""

    def test_dims_unchanged_after_construction(self, model_cls, cfg_factory):
        """Config dims must not be mutated when building the model."""
        cfg = cfg_factory()
        original_dims = list(cfg.post_mp_layer.dims)
        model = model_cls(self.num_features, self.num_classes, cfg)
        assert cfg.post_mp_layer.dims == original_dims
        # Verify dim_out is wired correctly via a forward pass
        _, pred, _ = model(self.load_batch())
        assert pred.shape[-1] == self.num_classes

    def test_repeated_construction_stable_dims(self, model_cls, cfg_factory):
        """Simulates k-fold: constructing the model N times with the same config
        must not accumulate trailing dim_out values."""
        cfg = cfg_factory()
        original_dims = list(cfg.post_mp_layer.dims)
        for _ in range(5):
            model = model_cls(self.num_features, self.num_classes, cfg)
            _, pred, _ = model(self.load_batch())
            assert pred.shape[-1] == self.num_classes
        assert cfg.post_mp_layer.dims == original_dims


class TestSTNodeClassifier(BatchLoaderMixin):
    def test_without_pooling(self):
        cfg = NodeClassifierModelConfig(
            mp_layers=[
                MessagePassingConfig(layer_type="gcnconv", pooling=None),
                MessagePassingConfig(layer_type="gcnconv", pooling=None),
            ],
            post_mp_layer=PostMPConfig(dims=[64, 32]),
        )
        batch = self.load_batch()
        model = STNodeClassifier(self.num_features, self.num_classes, cfg).to(
            self.device
        )
        batch, pred, other_loss = model(batch)
        assert isinstance(pred, Tensor)
        assert isinstance(batch, Data)
        assert pred.shape == (
            self.batch_size * self.num_nodes_per_graph,
            self.num_classes,
        )
