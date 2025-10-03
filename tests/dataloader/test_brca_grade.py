from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.brca_grade import BRCAGradeDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns
    data = {
        # ID_COL
        "cellID": ["cell1", "cell2", "cell3"],
        # GROUP_COLS
        "gender": ["FEMALE", "FEMALE", "FEMALE"],
        "age": [50, 50, 50],
        "PTNM_M": [0, 0, 0],
        "PTNM_T": ["1c", "1c", "1c"],
        "PTNM_N": [0, 0, 0],
        "Patientstatus": ["alive", "alive", "alive"],
        "diseasestatus": ["tumor", "tumor", "tumor"],
        "Post-surgeryTx": ["Chemotherapy", "Chemotherapy", "Chemotherapy"],
        "clinical_type": ["HR+HER2-", "HR+HER2-", "HR+HER2-"],
        # POS_COLS
        "X_centroid": [1.0, 2.0, 3.0],
        "Y_centroid": [1.0, 2.0, 3.0],
        # Feature columns (32 marker genes)
        "Fibronectin(Nd142)": [0.1, 0.2, 0.3],
        "SMA(Nd148)": [0.4, 0.5, 0.6],
        "Vimentin(Sm149)": [0.7, 0.8, 0.9],
        "CD68(Nd146)": [1.0, 1.1, 1.2],
        "CD3(Sm152)": [1.3, 1.4, 1.5],
        "CD44(Gd160)": [1.6, 1.7, 1.8],
        "CD45(Dy162)": [1.9, 2.0, 2.1],
        "Cytokeratin 7(Yb174)": [2.2, 2.3, 2.4],
        "EpCAM(Dy161)": [2.5, 2.6, 2.7],
        "Cytokeratin 19(Nd143)": [2.8, 2.9, 3.0],
        "pan Cytokeratin(Lu175)": [3.1, 3.2, 3.3],
        "Cytokeratin 8/18(Nd144)": [3.4, 3.5, 3.6],
        "GATA3(Dy163)": [3.7, 3.8, 3.9],
        "Rabbit IgG H L(Gd156)": [4.0, 4.1, 4.2],
        "Progesterone Receptor A/B'(Gd158)": [4.3, 4.4, 4.5],
        "mTOR(Yb173)": [4.6, 4.7, 4.8],
        "c-erbB-2 - Her2(Eu151)": [4.9, 5.0, 5.1],
        "Histone H3(In113)": [5.2, 5.3, 5.4],
        "S6(Er170)": [5.5, 5.6, 5.7],
        "Twist(Nd145)": [5.8, 5.9, 6.0],
        "CD31(Yb172)": [6.1, 6.2, 6.3],
        "vWF(Yb172)": [6.4, 6.5, 6.6],
        "CD20(Dy164)": [6.7, 6.8, 6.9],
        "Carbonic Anhydrase IX(Er166)": [7.0, 7.1, 7.2],
        "p53(Tb159)": [7.3, 7.4, 7.5],
        "EGFR(Tm169)": [7.6, 7.7, 7.8],
        "c-Myc(Nd150)": [7.9, 8.0, 8.1],
        "Slug(Gd155)": [8.2, 8.3, 8.4],
        "Ki-67(Er168)": [8.5, 8.6, 8.7],
        "Keratin 14 (KRT14)(Sm147)": [8.8, 8.9, 9.0],
        "Cytokeratin 5(Pr141)": [9.1, 9.2, 9.3],
        "DNA1(Ir191)": [9.4, 9.5, 9.6],
        # LABEL_COL
        "grade": [2, 2, 2],
    }
    return pd.DataFrame(data)


def test_brca_grade_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/brca-grade-test")
        ds = BRCAGradeDataset(root=data_root)
        assert len(ds) == 1, f"There should be only one graph sample but got {len(ds)}"
        data = ds[0]
        assert data.x.shape == (3, 32), "Shape of x should be (3 cells, 32 features)"
        assert data.y.item() == 1, "Label should be 1 (grade 2 -> 0-indexed as 1)"
        assert data.pos.shape == (3, 2), "Shape of pos should be (3 positions (x,y))"

        rm_dir_if_exists(data_root / "processed")
