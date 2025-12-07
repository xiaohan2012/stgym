import torch
from torch import Tensor
from torch_geometric.data import Data

from stgym.config_schema import GraphClassifierModelConfig, NodeClassifierModelConfig
from stgym.global_pooling import get_pooling_operator
from stgym.layers import MLP, GeneralMultiLayer


def check_edge_index(batch):
    if batch.edge_index.shape[1] == 0:
        raise ValueError(
            "There are no edges in the graph batch. If you're constructing the graph using radius, it is possible that the radius value is too small. Consider increasing it."
        )


class STGraphClassifier(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, cfg: GraphClassifierModelConfig):
        super().__init__()
        self.mp_module = GeneralMultiLayer(
            dim_in=dim_in, layer_configs=cfg.mp_layers, mem_config=cfg.mem
        )
        self.global_pooling = get_pooling_operator(cfg.global_pooling)

        # dim_out should be passed to MLP
        cfg.post_mp_layer.dims.append(dim_out)
        # TODO: what does it mean for dim equal -1? when to use it?
        self.post_mp = MLP(-1, cfg.post_mp_layer, cfg.mem)

    def forward(self, batch: Data) -> tuple[Data, Tensor, list[dict[str, Tensor]]]:
        """return the batch before global pooling and the predictions by post MP layers"""
        check_edge_index(batch)

        batch = self.mp_module(batch)
        if hasattr(batch, "loss"):
            other_loss = batch.loss
        else:
            other_loss = []
        graph_embeddings = self.global_pooling(batch.x, batch.batch)
        pred = self.post_mp(graph_embeddings)
        return batch, pred.squeeze(), other_loss


class STNodeClassifier(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, cfg: NodeClassifierModelConfig):
        super().__init__()
        self.mp_module = GeneralMultiLayer(
            dim_in=dim_in, layer_configs=cfg.mp_layers, mem_config=cfg.mem
        )
        # dim_out should be passed to MLP
        cfg.post_mp_layer.dims.append(dim_out)
        # TODO: what does it mean for dim equal -1? when to use it?
        self.post_mp = MLP(-1, cfg.post_mp_layer, cfg.mem)

    def forward(self, batch: Data) -> tuple[Data, Tensor, list[dict[str, Tensor]]]:
        check_edge_index(batch)

        batch = self.mp_module(batch)

        if hasattr(batch, "loss"):
            other_loss = batch.loss
        else:
            other_loss = []
        # TODO: we need to have some conditioning here
        # if pooling is used, we use the clustering vectors as the input features to post mp layers
        # otherwise, we use batch.x
        if hasattr(batch, "s"):
            s = torch.softmax(batch.s, axis=1)
            pred = self.post_mp(s)
        else:
            pred = self.post_mp(batch.x)
        # TODO: is batch still needed?
        return batch, pred.squeeze(), other_loss
