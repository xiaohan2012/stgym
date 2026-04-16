"""Tests for the Marimo notebook rct_experiment_analysis.py."""

from unittest import mock

import pandas as pd

from rct_experiment_analysis import app
from stgym.rct_utils import DESIGN_DIM_TO_MLFLOW_PATH
from tests.mock_mlflow import MockRun, MockRunData, MockRunInfo

_POOLING_DIM = "model.pooling.type"
_POOLING_PARAM = DESIGN_DIM_TO_MLFLOW_PATH[_POOLING_DIM]


class TestMarimoNotebook:
    """Tests for the Marimo notebook via programmatic execution."""

    @property
    def mock_runs(self) -> list[MockRun]:
        """Create mock runs: 3 groups × 2 design choices = 6 runs."""
        runs = []
        for group_id in range(3):
            for idx, design_choice in enumerate(["dmon", "mincut"]):
                metric = 0.75 + group_id * 0.02 + idx * 0.05
                runs.append(
                    MockRun(
                        data=MockRunData(
                            tags={
                                "group_id": f"group_{group_id}",
                                "design_dimension": _POOLING_DIM,
                            },
                            metrics={"test_roc_auc": metric},
                            params={_POOLING_PARAM: design_choice},
                        ),
                        info=MockRunInfo(status="FINISHED"),
                    )
                )
        return runs

    @mock.patch("stgym.rct_utils.fetch_runs")
    def test_notebook_runs_and_populates_results(self, mock_fetch_runs):
        """Verify notebook runs end-to-end and populates results_df with analysis."""
        mock_fetch_runs.return_value = self.mock_runs

        _outputs, variables = app.run()

        assert "results_df" in variables
        results_df = variables["results_df"]
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 6
        assert "rank" in results_df.columns
        assert "design_dimension" in results_df.columns
        assert _POOLING_DIM in results_df["design_dimension"].values
