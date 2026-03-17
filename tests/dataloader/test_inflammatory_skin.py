from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse

from stgym.data_loader.inflammatory_skin import N_TOP_GENES, InflammatorySkinDataset


def _create_mock_h5(path: Path, n_genes: int = 100):
    """Create a mock HDF5 file with 2 specimens (3 cells each)."""
    n_cells = 6
    specimens = np.array([0, 0, 0, 1, 1, 1])
    biopsy_types = np.array([1, 1, 1, 0, 0, 0])  # LESIONAL, NON LESIONAL
    spatial_coords = np.array(
        [[10, 20], [11, 21], [12, 22], [30, 40], [31, 41], [32, 42]],
        dtype=np.float32,
    )

    np.random.seed(42)
    gene_expression = np.random.rand(n_cells, n_genes).astype(np.float32)
    sparse_matrix = scipy.sparse.csr_matrix(gene_expression)

    with h5py.File(path, "w") as f:
        f.create_group("obsm").create_dataset("spatial", data=spatial_coords)
        obs = f.create_group("obs")
        obs.create_dataset("patient", data=specimens)
        obs.create_dataset("specimen", data=specimens)
        obs.create_dataset("biopsy_type", data=biopsy_types)
        cats = obs.create_group("__categories")
        cats.create_dataset(
            "biopsy_type",
            data=[b"NON LESIONAL", b"LESIONAL"],
            dtype=h5py.string_dtype(),
        )
        x = f.create_group("X")
        x.create_dataset("data", data=sparse_matrix.data)
        x.create_dataset("indices", data=sparse_matrix.indices)
        x.create_dataset("indptr", data=sparse_matrix.indptr)
        x.attrs["shape"] = sparse_matrix.shape


class TestInflammatorySkinDataset:
    def _make_dataset(self, tmp_path: Path, n_genes: int = 100):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _create_mock_h5(raw_dir / "source.h5", n_genes=n_genes)
        return InflammatorySkinDataset(root=tmp_path)

    def test_basic_properties(self, tmp_path):
        ds = self._make_dataset(tmp_path)

        assert len(ds) == 2
        for i in range(2):
            data = ds[i]
            assert data.x.shape == (3, 100)
            assert data.pos.shape == (3, 2)
            assert data.y.item() in [0, 1]

        labels = {ds[i].y.item() for i in range(len(ds))}
        assert labels == {0, 1}

    def test_hvg_filtering(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_genes=3000)
        assert ds[0].x.shape[1] == N_TOP_GENES

    def test_no_filtering_when_below_threshold(self, tmp_path):
        ds = self._make_dataset(tmp_path, n_genes=100)
        assert ds[0].x.shape[1] == 100

    def test_empty_file_raises(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        with h5py.File(raw_dir / "source.h5", "w"):
            pass

        with pytest.raises((KeyError, ValueError)):
            InflammatorySkinDataset(root=tmp_path)
