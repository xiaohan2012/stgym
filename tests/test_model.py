from torch import Tensor

from stgym.config_schema import (
    MessagePassingConfig,
    ModelConfig,
    PoolingConfig,
    PostMPConfig,
)
from stgym.model import GraphClassifier

from .utils import BatchLoaderMixin


class TestGraphClassifier(BatchLoaderMixin):
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
        model = GraphClassifier(self.num_features, self.num_classes, cfg)
        output = model(batch)
        assert isinstance(output, Tensor)
        assert output.shape == (self.batch_size, self.num_classes)
