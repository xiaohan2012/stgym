import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear as Linear_pyg

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from stgym.config_schema import MessagePassingConfig, LayerConfig, Config, MemoryConfig
from stgym.activation import get_activation_function

def get_layer_class(name):
    if name == 'gcnconv':
        return GCNConv
    elif name == 'sageconv':
        return SAGEConv
    elif name == 'ginconv':
        return GINConv
    elif name == 'linear':
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
    def __init__(self, dim_in:int, dim_out:int, mp_config: MessagePassingConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(
            dim_in,
            dim_out,
            bias=mp_config.has_bias,
        )

    def forward(self, batch):
        # TODO: why modify the x in place?
        batch.x = self.model(batch.x, batch.edge_index)
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
        batch.x = self.model(batch.x, batch.edge_index)
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
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

class GeneralLayer(torch.nn.Module):
    r"""A general wrapper for layers."""
    def __init__(self, name, dim_in:int, dim_out:int, layer_config: LayerConfig, mem_config: MemoryConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.l2norm

        self.layer = get_layer_class(name)(dim_in, dim_out, layer_config, **kwargs)
        layer_wrapper = []
        if layer_config.use_batchnorm:
            layer_wrapper.append(
                torch.nn.BatchNorm1d(
                    dim_out,
                    eps=layer_config.bn_eps,
                    momentum=layer_config.bn_momentum,
                ))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                torch.nn.Dropout(
                    p=layer_config.dropout,
                    inplace=mem_config.inplace,
                ))
        if layer_config.has_act:
            layer_wrapper.append(get_activation_function(layer_config.act))
        self.post_layer = torch.nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch
