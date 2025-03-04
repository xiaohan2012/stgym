import math

import torch

from stgym.config_schema import PoolingConfig
from stgym.pooling.dmon import DMoNPooling, DMoNPoolingLayer

from ..utils import BatchLoaderMixin


class TestDmonPooling:
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)

    def create_data(self):
        x = torch.randn((self.batch_size, self.num_nodes, self.channels))
        adj = torch.ones((self.batch_size, self.num_nodes, self.num_nodes))
        mask = torch.randint(0, 2, (self.batch_size, self.num_nodes), dtype=torch.bool)
        return x, adj, mask

    def test(self):
        x, adj, mask = self.create_data()

        pool = DMoNPooling(self.num_clusters)
        assert str(pool) == "DMoNPooling(-1, num_clusters=10)"

        s, x, adj, spectral_loss, ortho_loss, cluster_loss = pool(x, adj, mask)
        assert s.size() == (self.batch_size, self.num_nodes, self.num_clusters)
        assert x.size() == (self.batch_size, self.num_clusters, self.channels)
        assert adj.size() == (self.batch_size, self.num_clusters, self.num_clusters)
        assert -1 <= spectral_loss <= 0.5
        assert 0 <= ortho_loss <= math.sqrt(2)
        assert 0 <= cluster_loss <= math.sqrt(self.num_clusters) - 1


class TestWrapper(BatchLoaderMixin):
    def test(self):
        num_clusters = 10
        batch = self.load_batch()
        cfg = PoolingConfig(
            type="dmon",
            n_clusters=num_clusters,
        )
        # (self.out_channels, self.in_channels): ([10], 128)
        model = DMoNPoolingLayer(cfg)
        output_batch = model(batch)
        assert output_batch.x.shape == (
            self.batch_size,
            num_clusters,
            self.num_features,
        )
        assert output_batch.adj.shape == (self.batch_size, num_clusters, num_clusters)
