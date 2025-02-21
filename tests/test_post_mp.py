import pytest

from stgym.config_schema import Config
from stgym.post_mp import GNNGraphHead

from .utils import BatchLoaderMixin


class Test(BatchLoaderMixin):
    @pytest.mark.parametrize("graph_pooling", ["sum", "mean", "max"])
    def test(self, graph_pooling):
        # ensure that pooling runs and the output dimenion matchs
        # e.g., 2 graphs should output 2xnum_classes for graph classification
        config = Config(
            post_mp={
                "graph_pooling": graph_pooling,
                "n_layers": 2,
                "dim_inner": 64,
                "l2norm": False,
            },
            mp={},
            mem={},
            inter_layer={},
        )

        batch = self.load_batch()
        model = GNNGraphHead(self.num_features, self.num_classes, config)

        pred, label = model(batch)
        assert pred.shape == (self.batch_size, self.num_classes)

    # test mincut pool
    def test_mincut_pooling(self):
        pass
