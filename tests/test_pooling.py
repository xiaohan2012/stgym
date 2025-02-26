from stgym.config_schema import PoolingConfig
from stgym.pooling import DMoNPoolingLayer


class TestDMoNPoolingLayer:
    def test(self):
        cfg = PoolingConfig(type="dmon", n_clusters=10)
        layer = DMoNPoolingLayer(cfg)
        layer
