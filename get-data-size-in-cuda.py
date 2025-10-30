import torch
from torch_geometric.data import Data

data = Data(
    x=torch.randn(1000, 128),
    edge_index=torch.randint(0, 1000, (2, 5000)),
    edge_attr=torch.randn(5000, 16),
    y=torch.randint(0, 10, (1000,)),
)


def get_cuda_memory_usage(data):
    """Get GPU memory usage for Data object."""
    if not torch.cuda.is_available():
        return 0

    total_bytes = 0
    for key, item in data:
        if torch.is_tensor(item) and item.is_cuda:
            total_bytes += item.element_size() * item.nelement()

    return total_bytes / (1024 * 1024)


# Move data to GPU
data = data.to("cuda")
gpu_size = get_cuda_memory_usage(data)
print(f"GPU memory usage: {gpu_size:.2f} MB")
