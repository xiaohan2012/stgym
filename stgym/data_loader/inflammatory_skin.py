from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from .base import AbstractDataset

RAW_FILE_NAME = "source.h5"
POS_KEY = "obsm/spatial"
GROUP_COL = "specimen"  # Changed from patient to specimen
LABEL_COL = "biopsy_type"
FEATURE_KEY = "X"


class InflammatorySkinDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        h5_data_path = Path(self.raw_dir) / RAW_FILE_NAME

        with h5py.File(h5_data_path, "r") as f:
            # Load spatial coordinates
            spatial_coords = f[POS_KEY][:]  # Shape: (n_cells, 2)

            # Load specimen IDs and biopsy types
            specimens = f["obs"]["specimen"][:]  # Encoded specimen IDs
            biopsy_types = f["obs"]["biopsy_type"][:]  # Encoded biopsy types

            # Get the mapping for biopsy types (LESIONAL, NON LESIONAL)
            biopsy_categories = f["obs"]["__categories"]["biopsy_type"][:]
            biopsy_map = {
                i: cat.decode() if hasattr(cat, "decode") else str(cat)
                for i, cat in enumerate(biopsy_categories)
            }

            # Load gene expression matrix (sparse format)
            gene_expression = self._load_sparse_matrix(f[FEATURE_KEY])

            # Convert to dense for easier processing
            if hasattr(gene_expression, "toarray"):
                gene_expression = gene_expression.toarray()

            # AnnData stores as (n_genes, n_cells), we need (n_cells, n_genes)
            # Always transpose from AnnData format
            gene_expression = gene_expression.T

            # Group cells by specimen to create graphs
            unique_specimens = np.unique(specimens)
            data_list = []

            for specimen_id in unique_specimens:
                # Get cells for this specimen
                specimen_mask = specimens == specimen_id
                specimen_cells = np.where(specimen_mask)[0]

                if len(specimen_cells) == 0:
                    continue

                # Extract specimen data
                specimen_pos = torch.tensor(
                    spatial_coords[specimen_cells], dtype=torch.float
                )
                specimen_features = torch.tensor(
                    gene_expression[specimen_cells], dtype=torch.float
                )

                # Get label for this specimen (should be consistent across all cells)
                specimen_biopsy_types = biopsy_types[specimen_mask]
                unique_biopsy_types = np.unique(specimen_biopsy_types)

                if len(unique_biopsy_types) != 1:
                    raise ValueError(
                        f"Specimen {specimen_id} has inconsistent biopsy types: "
                        f"{[biopsy_map[bt] for bt in unique_biopsy_types]}"
                    )

                biopsy_type_encoded = unique_biopsy_types[0]
                biopsy_type_name = biopsy_map[biopsy_type_encoded]

                # Convert to binary label: LESIONAL=1, NON LESIONAL=0
                if biopsy_type_name == "LESIONAL":
                    label = 1
                elif biopsy_type_name == "NON LESIONAL":
                    label = 0
                else:
                    raise ValueError(f"Unexpected biopsy type: {biopsy_type_name}")

                y = torch.tensor(label, dtype=torch.long)

                # Create Data object
                data = Data(x=specimen_features, y=y, pos=specimen_pos)
                data_list.append(data)

        return data_list

    def _load_sparse_matrix(self, sparse_group):
        """Load sparse matrix from HDF5 group (AnnData format)."""
        try:
            # Try to load as sparse matrix
            data = sparse_group["data"][:]
            indices = sparse_group["indices"][:]
            indptr = sparse_group["indptr"][:]
            shape = sparse_group.attrs.get("shape", None)

            if shape is None:
                # Try to infer shape
                n_rows = len(indptr) - 1
                n_cols = max(indices) + 1 if len(indices) > 0 else 0
                shape = (n_rows, n_cols)

            # Create sparse matrix
            import scipy.sparse

            sparse_matrix = scipy.sparse.csr_matrix(
                (data, indices, indptr), shape=shape
            )
            return sparse_matrix

        except (KeyError, AttributeError):
            # Fallback: try to load as dense array
            try:
                return sparse_group[:]
            except Exception as e:
                raise ValueError(f"Could not load matrix data: {e}")
