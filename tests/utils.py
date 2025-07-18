import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def create_synthetic_data(
    num_nodes: int = 100, num_features: int = 128, num_classes: int = 10
):
    # Generate random node features
    x = torch.randn((num_nodes, num_features))

    # Generate random edge indices
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

    # Generate random labels
    y = torch.randint(0, num_classes, (num_nodes,))

    return Data(x=x, edge_index=edge_index, y=y)


def create_data_batch(
    num_nodes_per_graph: int = 100,
    num_features: int = 128,
    num_classes: int = 10,
    batch_size: int = 5,
    device: str = "cpu",
):
    dataloader = DataLoader(
        [
            create_synthetic_data(num_nodes_per_graph, num_features, num_classes)
            for _ in range(batch_size)
        ],
        batch_size=batch_size,
    )

    batch = next(iter(dataloader))

    adj = torch.zeros(
        (num_nodes_per_graph * batch_size, num_nodes_per_graph * batch_size)
    )
    adj[batch.edge_index[0], batch.edge_index[1]] = 1
    batch.adj_t = adj.to_sparse_coo()
    # graph-level labels
    batch.y = torch.randint(0, 2, (batch_size,))
    return batch.to(device)


DEVICE = "cpu" if not torch.cuda.is_available() else "cuda:0"


class BatchLoaderMixin:
    num_nodes_per_graph = 100
    num_features = 128
    num_classes = 10
    batch_size = 3
    device = DEVICE

    def load_batch(self):
        return create_data_batch(
            num_nodes_per_graph=self.num_nodes_per_graph,
            num_features=self.num_features,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            device=self.device,
        )


RANDOM_SEEDS = [42, 123, 9999, 341324]
