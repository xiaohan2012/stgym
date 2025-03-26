import torch
import torch.nn.functional as F

from stgym.config_schema import ModelConfig
from stgym.layers import GeneralLayer


class GNNStackStage(torch.nn.Module):
    r"""Stacks a number of GNN layers.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
    """

    def __init__(self, dim_in: int, dim_out: int, config: ModelConfig):
        super().__init__()
        self.num_layers = config.mp.n_layers
        self.stage_type = config.inter_layer.stage_type
        self.l2_norm = config.mp.l2norm
        dim_inner = config.mp.dim_inner
        for i in range(self.num_layers):
            if self.stage_type == "skipconcat":
                d_in = dim_in if i == 0 else dim_in + i * dim_inner
                d_out = dim_out if i == self.num_layers - 1 else dim_inner
            else:
                d_in = dim_in if i == 0 else dim_inner
                d_out = dim_out if i == self.num_layers - 1 else dim_inner
            # TODO: the dimension calculation should be corrected, current code does not run
            layer = GeneralLayer(
                config.mp.layer_type, d_in, d_out, config.mp, config.mem
            )
            self.add_module(f"layer{i}", layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if self.stage_type == "skipsum":
                batch.x = x + batch.x
            elif self.stage_type == "skipconcat" and i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)
        if self.l2_norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch
