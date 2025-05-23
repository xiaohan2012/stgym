import torch

from stgym.config_schema import GraphClassifierModelConfig
from stgym.layers import MLP
from stgym.pooling import get_pooling_function


class GNNGraphHead(torch.nn.Module):
    r"""A GNN prediction head for graph-level prediction tasks.
    A post message passing layer (as specified by :obj:`cfg.gnn.post_mp`) is
    used to transform the pooled graph-level embeddings using an MLP.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
    """

    def __init__(self, dim_in: int, dim_out: int, config: GraphClassifierModelConfig):
        super().__init__()
        self.layer_post_mp = MLP(dim_in, dim_out, config.post_mp, config.mem)
        self.graph_pooling_type = config.post_mp.graph_pooling
        self.pooling_fun = get_pooling_function(self.graph_pooling_type)

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        if self.graph_pooling_type == "mincut":
            graph_emb = self.pooling_fun(batch.x, batch.adj_t, batch.s)
        else:
            graph_emb = self.pooling_fun(batch.x, batch.batch)

        graph_emb = self.layer_post_mp(graph_emb)

        # TODO: what should forward return?
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
