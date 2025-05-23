import torch
from torch import Tensor
from torch_geometric.data import Data

from stgym.config_schema import (
    ClusteringModelConfig,
    MessagePassingConfig,
    ModelConfig,
    PoolingConfig,
    PostMPConfig,
)
from stgym.model import STClusteringModel, STGraphClassifier

from .utils import BatchLoaderMixin


class TestSTGraphClassifier(BatchLoaderMixin):
    def test(self):
        cfg = ModelConfig(
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
        model = STGraphClassifier(self.num_features, self.num_classes, cfg)
        batch, pred, other_loss = model(batch)
        assert isinstance(pred, Tensor)
        assert isinstance(batch, Data)
        assert pred.shape == (self.batch_size, self.num_classes)
        assert len(other_loss) == 2
        for loss in other_loss:
            assert isinstance(loss, dict)


class TestSTClusteringModel(BatchLoaderMixin):
    def test(self):
        cfg = ClusteringModelConfig(
            mp_layers=[
                MessagePassingConfig(
                    layer_type="gcnconv",
                    pooling=PoolingConfig(type="dmon", n_clusters=self.num_classes),
                ),
            ],
        )
        batch = self.load_batch()
        model = STClusteringModel(self.num_features, cfg)
        batch, pred, loss = model(batch)
        assert isinstance(pred, Tensor)
        assert torch.allclose(
            pred.sum(axis=1),
            torch.ones(self.num_nodes_per_graph * self.batch_size),
            rtol=1e-5,
        )
        assert isinstance(batch, Data)
        assert pred.shape == (
            self.batch_size * self.num_nodes_per_graph,
            self.num_classes,
        )
        assert len(loss) == 1
        assert isinstance(loss[0], dict)
