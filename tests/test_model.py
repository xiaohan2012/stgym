from torch import Tensor

from stgym.config_schema import (
    MessagePassingConfig,
    ModelConfig,
    PoolingConfig,
    PostMPConfig,
)
from stgym.model import STGraphClassifier
from torch_geometric.data import Data

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
            isinstance(loss, dict)
