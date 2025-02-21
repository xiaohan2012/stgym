import pytest

from stgym.config_schema import Config
from stgym.mp import GNNStackStage

from .utils import BatchLoaderMixin


class TestGNNStackStage(BatchLoaderMixin):
    @pytest.mark.parametrize("stage_type", ["skipconcat", "skipconcat", None])
    def test(self, stage_type):
        config = Config(
            mp={
                "layer_type": "gcnconv",
                "n_layers": 2,
                "dim_inner": 64,
                "l2norm": False,
            },
            post_mp={},
            mem={},
            inter_layer={"stage_type": stage_type},
        )

        batch = self.load_batch()
        model = GNNStackStage(self.num_features, self.num_classes, config)

        output = model(batch)
        assert output.x.shape == (self.num_nodes * self.batch_size, self.num_classes)
