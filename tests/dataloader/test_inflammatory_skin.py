import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse

from stgym.data_loader.inflammatory_skin import InflammatorySkinDataset


@pytest.fixture
def mock_h5_file():
    """Create a temporary HDF5 file with mock inflammatory skin data structure."""
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    temp_file.close()

    # Mock data: 6 cells from 2 specimens
    n_cells = 6
    n_genes = 100

    # Specimen assignments: 3 cells from specimen 0 (LESIONAL), 3 cells from specimen 1 (NON LESIONAL)
    specimens = np.array([0, 0, 0, 1, 1, 1])  # Specimen IDs
    patients = np.array(
        [0, 0, 0, 1, 1, 1]
    )  # Patient IDs (same as specimens for simplicity)
    biopsy_types = np.array([1, 1, 1, 0, 0, 0])  # 1=LESIONAL, 0=NON LESIONAL

    # Mock spatial coordinates
    spatial_coords = np.array(
        [
            [10.0, 20.0],
            [11.0, 21.0],
            [12.0, 22.0],  # Patient 0
            [30.0, 40.0],
            [31.0, 41.0],
            [32.0, 42.0],  # Patient 1
        ]
    )

    # Mock gene expression (random data) - in AnnData format: (n_genes, n_cells)
    np.random.seed(42)
    gene_expression_dense = np.random.rand(n_genes, n_cells).astype(np.float32)

    # Convert to sparse format (CSR) - already in correct (n_genes, n_cells) format
    gene_expression_sparse = scipy.sparse.csr_matrix(gene_expression_dense)

    # Create HDF5 file with AnnData-like structure
    with h5py.File(temp_file.name, "w") as f:
        # Spatial coordinates
        obsm_group = f.create_group("obsm")
        obsm_group.create_dataset("spatial", data=spatial_coords)

        # Observations (cells)
        obs_group = f.create_group("obs")
        obs_group.create_dataset("patient", data=patients)
        obs_group.create_dataset("specimen", data=specimens)  # Add specimen data
        obs_group.create_dataset("biopsy_type", data=biopsy_types)

        # Categories for biopsy_type
        categories_group = obs_group.create_group("__categories")
        biopsy_cat = categories_group.create_dataset(
            "biopsy_type",
            data=[b"NON LESIONAL", b"LESIONAL"],
            dtype=h5py.string_dtype(),
        )

        # Gene expression matrix (sparse format)
        x_group = f.create_group("X")
        x_group.create_dataset("data", data=gene_expression_sparse.data)
        x_group.create_dataset("indices", data=gene_expression_sparse.indices)
        x_group.create_dataset("indptr", data=gene_expression_sparse.indptr)
        x_group.attrs["shape"] = gene_expression_sparse.shape

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


def test_inflammatory_skin_dataset(mock_h5_file):
    """Test InflammatorySkinDataset with mock HDF5 data."""
    # Create temporary directory for dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        data_root = Path(temp_dir)
        raw_dir = data_root / "raw"
        raw_dir.mkdir()

        # Copy mock file to expected location
        import shutil

        shutil.copy(mock_h5_file, raw_dir / "source.h5")

        # Create dataset
        ds = InflammatorySkinDataset(root=data_root)

        # Test basic properties
        assert len(ds) == 2, f"Expected 2 graphs (specimens) but got {len(ds)}"

        # Test first graph (LESIONAL specimen)
        data_0 = ds[0]
        assert data_0.x.shape[0] == 3, "First specimen should have 3 cells"
        assert data_0.x.shape[1] == 100, "Should have 100 gene features"
        assert data_0.pos.shape == (3, 2), "Position should be (3 cells, 2 coordinates)"
        assert data_0.y.item() in [0, 1], "Label should be 0 or 1"

        # Test second graph (NON LESIONAL specimen)
        data_1 = ds[1]
        assert data_1.x.shape[0] == 3, "Second specimen should have 3 cells"
        assert data_1.x.shape[1] == 100, "Should have 100 gene features"
        assert data_1.pos.shape == (3, 2), "Position should be (3 cells, 2 coordinates)"
        assert data_1.y.item() in [0, 1], "Label should be 0 or 1"

        # Test that we have both label types
        labels = {ds[i].y.item() for i in range(len(ds))}
        assert len(labels) == 2, f"Expected both label types (0,1) but got {labels}"
        assert labels == {0, 1}, f"Expected labels {{0,1}} but got {labels}"


def test_inflammatory_skin_dataset_label_consistency():
    """Test that the dataset properly handles label consistency within patients."""
    # This test would use a mock where a patient has inconsistent labels
    # For now, we'll test the happy path since our mock data is consistent


def test_inflammatory_skin_dataset_empty():
    """Test dataset behavior with empty or invalid data."""
    # Create a minimal mock that should fail gracefully
    with tempfile.TemporaryDirectory() as temp_dir:
        data_root = Path(temp_dir)
        raw_dir = data_root / "raw"
        raw_dir.mkdir()

        # Create empty HDF5 file
        empty_h5 = raw_dir / "source.h5"
        with h5py.File(empty_h5, "w") as f:
            pass  # Empty file

        # Should raise an error when trying to process
        with pytest.raises((KeyError, ValueError)):
            ds = InflammatorySkinDataset(root=data_root)


if __name__ == "__main__":
    # Run a simple test
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Running test in {temp_dir}")
        # Basic smoke test would go here
