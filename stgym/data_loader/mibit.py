from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.data.makedirs import makedirs


@dataclass(frozen=True)
class MIBITNBCSourceInfo:
    download_url_suffix = "https://raw.githubusercontent.com/huBioinfo/CytoCommunity/main/Tutorial/Supervised/MIBI-TNBC_Input"
    patient_ids = sorted(
        [f"patient{i}" for i in (set(range(1, 42)) - {15, 19, 22, 24, 25, 26, 30})]
    )
    # patient_ids = [1]
    cell_type_suffix = "CellTypeLabel"
    cell_coordinates_suffix = "Coordinates"
    graph_label_suffix = "GraphLabel"


class MIBITNBCDataSet(InMemoryDataset):
    """
    example usage:

    ds = MIBITNBCDataSet(root="./data/MIBITNBC")

    or build the edges using k-nearest-neighbours

    import torch_geometric.transforms as T
    ds = MIBITNBCDataSet(root='./data/MIBITNBC/', transform=T.KNNGraph(k=5))
    """

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        train_val_test_split_ratios=[0.7, 0.1, 0.2],
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

        assert np.isclose(
            sum(train_val_test_split_ratios), 1.0
        ), f"{sum(train_val_test_split_ratios)} != 1"

        np.random.seed(42)

        N = len(self)
        ratios = np.array([0.7, 0.1, 0.2])
        split_indices = np.split(
            np.random.permutation(N),
            np.round(np.cumsum(ratios * N))[: ratios.shape[0] - 1].astype(int),
        )
        self.data.train_graph_index = split_indices[0]
        self.data.val_graph_index = split_indices[1]
        self.data.test_graph_index = split_indices[2]

    @property
    def download_url_suffix(self):
        return MIBITNBCSourceInfo.download_url_suffix

    @property
    def cell_type_suffix(self):
        return MIBITNBCSourceInfo.cell_type_suffix

    @property
    def cell_coordinates_suffix(self):
        return MIBITNBCSourceInfo.cell_coordinates_suffix

    @property
    def graph_label_suffix(self):
        return MIBITNBCSourceInfo.graph_label_suffix

    @property
    def patient_ids_to_download(self):
        return MIBITNBCSourceInfo.patient_ids

    @property
    def raw_file_names(self):
        return self.patient_ids_to_download

    def get_cell_type_file_url(self, patient_id: str) -> str:
        return f"{self.download_url_suffix}/{patient_id}_{self.cell_type_suffix}.txt"

    def get_cell_coordinates_file_url(self, patient_id: str) -> str:
        return f"{self.download_url_suffix}/{patient_id}_{self.cell_coordinates_suffix}.txt"

    def get_graph_label_file_url(self, patient_id: str) -> str:
        return f"{self.download_url_suffix}/{patient_id}_{self.graph_label_suffix}.txt"

    def download(self) -> None:
        for pid in self.raw_file_names:
            download_url(self.get_cell_type_file_url(pid), self.raw_dir)
            download_url(self.get_cell_coordinates_file_url(pid), self.raw_dir)
            download_url(self.get_graph_label_file_url(pid), self.raw_dir)

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process_one_patient(self, patient_id: str) -> Data:
        """read the cell coordinates and graph label for patient with `patient_id` and return a Data object"""
        graph_label_file_path = Path(self.raw_dir) / Path(
            f"{patient_id}_{self.graph_label_suffix}.txt"
        )
        cell_coordinates_file_path = Path(self.raw_dir) / Path(
            f"{patient_id}_{self.cell_coordinates_suffix}.txt"
        )

        coordinates = pd.read_csv(
            cell_coordinates_file_path, sep="\t", header=None, names=["x", "y"]
        ).to_numpy()

        label = torch.tensor(
            pd.read_csv(graph_label_file_path, header=None).to_numpy()[0],
            dtype=torch.float,
        )
        print(f"label: {label}")
        return Data(y=label, pos=torch.Tensor(coordinates))

    def assign_x_to_graph(self, data_list: list[Data]) -> None:
        """assign the node-level cell types to each Data object in data_list by reading from the cell type files

        this modifies the given data list in place
        """
        column_name = "cell_type"

        cell_types_all_graphs = pd.DataFrame()
        num_nodes_per_graph = []
        for patient_id in self.raw_file_names:
            cell_type = pd.read_csv(
                Path(self.raw_dir) / Path(f"{patient_id}_{self.cell_type_suffix}.txt"),
                names=[column_name],
            )
            cell_types_all_graphs = pd.concat([cell_types_all_graphs, cell_type])
            num_nodes_per_graph.append(cell_type.shape[0])
        # the colated 1hot matrix
        x_1hot = torch.Tensor(
            pd.get_dummies(cell_types_all_graphs, columns=[column_name]).to_numpy()
        )

        # slice for each graph
        slices = np.concatenate([[0], np.cumsum(num_nodes_per_graph)])

        for i, data in enumerate(data_list):
            data.x = x_1hot[slices[i] : slices[i + 1], :]

    def process(self) -> list[Data]:
        """process and save data objects"""
        makedirs(self.processed_dir)
        data_list = []
        for patient_id in self.raw_file_names:
            data = self.process_one_patient(patient_id)
            data_list.append(data)
        self.assign_x_to_graph(data_list)
        self.save(data_list, self.processed_paths[0])

        return data_list
