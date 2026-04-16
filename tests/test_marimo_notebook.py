"""Tests for the Marimo notebook rct_experiment_analysis.py."""

from unittest import mock

import pandas as pd

from rct_experiment_analysis import app
from tests.conftest import POOLING_DIM


class TestMarimoNotebook:
    """Tests for the Marimo notebook via programmatic execution."""

    @mock.patch("stgym.rct_utils.fetch_runs")
    def test_notebook_runs_and_populates_results(self, mock_fetch_runs, regular_runs):
        """Verify notebook runs end-to-end and populates results_df with analysis."""
        mock_fetch_runs.return_value = regular_runs

        _outputs, variables = app.run()

        assert "results_df" in variables
        results_df = variables["results_df"]
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 6
        assert "rank" in results_df.columns
        assert "design_dimension" in results_df.columns
        assert POOLING_DIM in results_df["design_dimension"].values
