import torch
from torch import Tensor
from torch_geometric.data import Data

from stgym.config_schema import ModelConfig
from stgym.global_pooling import get_pooling_operator
from stgym.layers import MLP, GeneralMultiLayer


class STGraphClassifier(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, cfg: ModelConfig):
        super().__init__()
        self.mp_module = GeneralMultiLayer(
            dim_in=dim_in, layer_configs=cfg.mp_layers, mem_config=cfg.mem
        )
        self.global_pooling = get_pooling_operator(cfg.global_pooling)

        # dim_out should be passed to MLP
        cfg.post_mp_layer.dims.append(dim_out)
        # TODO: what does it mean for dim equal -1? when to use it?
        self.post_mp = MLP(-1, cfg.post_mp_layer, cfg.mem)

    def forward(self, batch: Data) -> Tensor:
        batch = self.mp_module(batch)
        graph_embeddings = self.global_pooling(batch.x, batch.batch)
        return self.post_mp(graph_embeddings)
