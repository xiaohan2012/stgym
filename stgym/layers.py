import pydash as _
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import Linear as Linear_pyg

from stgym.activation import get_activation_function
from stgym.config_schema import (
    LayerConfig,
    MemoryConfig,
    MessagePassingConfig,
    MultiLayerConfig,
    PostMPConfig,
)
from stgym.pooling import get_pooling_class
from stgym.utils import get_edge_weight


def get_layer_class(name):
    if name == "gcnconv":
        return GCNConv
    elif name == "sageconv":
        return SAGEConv
    elif name == "ginconv":
        return GINConv
    elif name == "linear":
        return Linear
    else:
        raise ValueError


class Linear(torch.nn.Module):
    r"""A basic Linear layer.

    Args:
        dim_in (int): The input dimension.
        dim_out (int): The output dimension.
        mp_config (MessagePassingConfig): The configuration of the message passing layer.
    """

    def __init__(self, dim_in: int, dim_out: int, mp_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = Linear_pyg(
            dim_in,
            dim_out,
            bias=mp_config.has_bias,
        )

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class GCNConv(torch.nn.Module):
    r"""A Graph Convolutional Network (GCN) layer."""

    def __init__(
        self, dim_in: int, dim_out: int, mp_config: MessagePassingConfig, **kwargs
    ):
        super().__init__()
        self.model = pyg.nn.GCNConv(
            dim_in,
            dim_out,
            bias=mp_config.has_bias,
        )

    def forward(self, batch):
        # TODO: why modify the x in place?
        batch.x = self.model(batch.x, batch.edge_index, get_edge_weight(batch))
        return batch


class SAGEConv(torch.nn.Module):
    r"""A GraphSAGE layer."""

    def __init__(self, dim_in: int, dim_out: int, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SAGEConv(
            dim_in,
            dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, get_edge_weight(batch))
        return batch


class GINConv(torch.nn.Module):
    r"""A Graph Isomorphism Network (GIN) layer."""

    def __init__(self, dim_in: int, dim_out: int, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = torch.nn.Sequential(
            Linear_pyg(dim_in, dim_out),  # Han: why not use dim_inner?
            torch.nn.ReLU(),
            Linear_pyg(dim_out, dim_out),
        )
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, get_edge_weight(batch))
        return batch


class GeneralLayer(torch.nn.Module):
    r"""A general wrapper for layers."""

    def __init__(
        self,
        layer_type: str,
        dim_in: int,
        dim_out: int,
        layer_config: LayerConfig | MessagePassingConfig | PostMPConfig,
        mem_config: MemoryConfig,
        **kwargs,
    ):
        super().__init__()
        self.has_l2norm = layer_config.l2norm

        self.layer = get_layer_class(layer_type)(
            dim_in, dim_out, layer_config, **kwargs
        )
        layer_wrapper = []
        if layer_config.use_batchnorm:
            layer_wrapper.append(
                torch.nn.BatchNorm1d(
                    dim_out,
                    eps=layer_config.bn_eps,
                    momentum=layer_config.bn_momentum,
                )
            )
        if layer_config.dropout > 0:
            layer_wrapper.append(
                torch.nn.Dropout(
                    p=layer_config.dropout,
                    inplace=mem_config.inplace,
                )
            )
        if layer_config.has_act:
            layer_wrapper.append(get_activation_function(layer_config.act))

        if self.should_apply_pooling(layer_config):
            self.pooling_layer = get_pooling_class(layer_config.pooling.type)(
                layer_config.pooling
            )
        else:
            self.pooling_layer = lambda _: _

        self.post_layer = torch.nn.Sequential(*layer_wrapper)

    def should_apply_pooling(self, layer_config):
        return (
            isinstance(layer_config, MessagePassingConfig) and layer_config.has_pooling
        )

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        elif isinstance(batch, pyg.data.Data):
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
            # apply pooling only if batch is not a Tensor
            batch = self.pooling_layer(batch)
        else:
            raise TypeError(f"Unsupported type: {type(batch)}")
        return batch


class GeneralMultiLayer(torch.nn.Module):
    r"""A general wrapper class for a stacking multiple NN layers."""

    def __init__(
        self,
        layer_type: str,
        dim_in: int,
        config: MultiLayerConfig,
        mem_config: MemoryConfig,
        **kwargs,
    ):
        super().__init__()
        dim_inner_list = _.map_(config.layers, "dim_inner")
        for i, layer_config in enumerate(config.layers):
            d_in = dim_in if i == 0 else dim_inner_list[i - 1]
            # d_out = dim_out if i == config.n_layers - 1 else dim_inner
            d_out = dim_inner_list[i]
            layer = GeneralLayer(
                layer_type, d_in, d_out, layer_config, mem_config, **kwargs
            )
            self.add_module(f"Layer_{i}", layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


class MLP(torch.nn.Module):
    """A basic MLP model."""

    def __init__(
        self,
        dim_in: int,
        multi_layer_config: MultiLayerConfig,
        mem_config: MemoryConfig,
        **kwargs,
    ):
        super().__init__()

        self.model = torch.nn.Sequential(
            GeneralMultiLayer(
                "linear",
                dim_in=dim_in,
                config=multi_layer_config,
                mem_config=mem_config,
            )
        )

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch
