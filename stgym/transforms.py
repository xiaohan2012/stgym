from typing import Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class AssignSparseCSC(BaseTransform):
    """Add Sparse CSC matrix to the data.

    Note: apply it after ToSparseTensor.
    """

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if "adj_t" not in store:
                continue
            store.adj_t_csc = store.adj_t.to_sparse_csc()
        return data


class AssignSparseCSR(BaseTransform):
    """Add Sparse CSR matrix to the data.

    Note: apply it after ToSparseTensor.
    """

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if "adj_t" not in store:
                continue
            store.adj_t_csr = store.adj_t.to_sparse_csr()
        return data
