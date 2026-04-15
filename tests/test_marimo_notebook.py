"""Tests for the Marimo notebook rct_experiment_analysis.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stgym.rct_utils import (
    aggregate_kfold_metrics,
    analyze_experiment,
    filter_complete_groups,
    runs_to_dataframe,
    summarize_ranks_by_design_choice,
)
from tests.mock_mlflow import MockRun, MockRunData, MockRunInfo


class TestMarimoNotebook:
    """Tests for the Marimo notebook validity and importability."""

    @property
    def notebook_path(self) -> Path:
        """Path to the Marimo notebook."""
        return Path(__file__).parent.parent / "rct_experiment_analysis.py"

    def test_marimo_check_passes(self):
        """Verify the notebook passes marimo check (syntax and structure)."""
        result = subprocess.run(
            [sys.executable, "-m", "marimo", "check", str(self.notebook_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"marimo check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_notebook_importable(self):
        """Verify the notebook can be imported as a Python module."""
        spec = importlib.util.spec_from_file_location(
            "rct_experiment_analysis", self.notebook_path
        )
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "app")
        assert hasattr(module.app, "run")


_POOLING_DIM = "model.pooling.type"
_POOLING_PARAM = "model/mp_layers/0/pooling/type"


@mock.patch("stgym.rct_utils.fetch_runs")
class TestAnalyzeExperimentWithRealisticData:
    """Test analyze_experiment with realistic mock data including k-fold runs."""

    @property
    def regular_runs(self) -> list[MockRun]:
        """Create mock runs without k-fold (standard single-run experiments).

        Creates 6 runs across 3 groups, each group has 2 design choices.
        """
        runs = []
        design_choices = ["dmon", "mincut"]

        for group_id in range(3):
            for idx, design_choice in enumerate(design_choices):
                metric = 0.75 + group_id * 0.02 + idx * 0.05
                run = MockRun(
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
                runs.append(run)

        return runs

    @property
    def kfold_runs(self) -> list[MockRun]:
        """Create mock runs with k-fold cross-validation.

        Creates 12 runs: 2 groups x 2 design choices x 3 folds.
        Each group shares the same group_id across folds.
        """
        runs = []
        design_choices = ["dmon", "mincut"]
        n_folds = 3

        for group_id in range(2):
            for design_choice in design_choices:
                for fold in range(n_folds):
                    base_metric = 0.80 + group_id * 0.03
                    if design_choice == "dmon":
                        base_metric += 0.02
                    metric = base_metric + (fold - 1) * 0.01

                    run = MockRun(
                        data=MockRunData(
                            tags={
                                "group_id": f"kfold_group_{group_id}_{design_choice}",
                                "fold": str(fold),
                                "design_dimension": _POOLING_DIM,
                            },
                            metrics={"test_roc_auc": metric},
                            params={_POOLING_PARAM: design_choice},
                        ),
                        info=MockRunInfo(status="FINISHED"),
                    )
                    runs.append(run)

        return runs

    @property
    def mixed_runs(self) -> list[MockRun]:
        """Create mock data with both regular and k-fold runs."""
        return self.regular_runs + self.kfold_runs

    def test_with_regular_runs_only(self, mock_fetch_runs: MagicMock):
        """Verify analysis works with regular (non-k-fold) runs."""
        mock_fetch_runs.return_value = self.regular_runs

        results = analyze_experiment(
            tracking_uri="http://localhost:5001",
            experiment_id="test_exp",
            metric_name="test_roc_auc",
            aggregate_kfold=True,
        )

        assert _POOLING_DIM in results
        result = results[_POOLING_DIM]
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6
        assert "rank" in result.columns
        assert set(result["design_choice"].unique()) == {"dmon", "mincut"}
        assert result["rank"].min() == 1.0
        assert result["rank"].max() == 2.0

    def test_with_kfold_runs_only(self, mock_fetch_runs: MagicMock):
        """Verify analysis correctly aggregates k-fold runs by mean metric."""
        mock_fetch_runs.return_value = self.kfold_runs

        results = analyze_experiment(
            tracking_uri="http://localhost:5001",
            experiment_id="test_exp",
            metric_name="test_roc_auc",
            aggregate_kfold=True,
        )

        assert _POOLING_DIM in results
        result = results[_POOLING_DIM]
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert result["fold"].isna().all()
        assert "rank" in result.columns

    def test_with_mixed_runs(self, mock_fetch_runs: MagicMock):
        """Verify analysis handles mixed regular and k-fold runs."""
        mock_fetch_runs.return_value = self.mixed_runs

        results = analyze_experiment(
            tracking_uri="http://localhost:5001",
            experiment_id="test_exp",
            metric_name="test_roc_auc",
            aggregate_kfold=True,
        )

        assert _POOLING_DIM in results
        result = results[_POOLING_DIM]
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "rank" in result.columns

    def test_kfold_aggregation_computes_mean(self, mock_fetch_runs: MagicMock):
        """Verify k-fold aggregation computes correct mean metrics."""
        mock_fetch_runs.return_value = self.kfold_runs

        df = runs_to_dataframe(self.kfold_runs, metric_name="test_roc_auc")

        aggregated = aggregate_kfold_metrics(df)

        assert len(aggregated) == 4

        for group_id in aggregated["group_id"].unique():
            group_df = df[df["group_id"] == group_id]
            expected_mean = group_df["metric"].mean()
            actual_mean = aggregated[aggregated["group_id"] == group_id]["metric"].iloc[
                0
            ]
            assert actual_mean == pytest.approx(expected_mean)

    def test_failed_fold_marks_group_failed(self, mock_fetch_runs: MagicMock):
        """Verify a failed fold marks the entire aggregated group as failed."""
        kfold_runs_with_failure = list(self.kfold_runs)
        kfold_runs_with_failure[0] = MockRun(
            data=kfold_runs_with_failure[0].data,
            info=MockRunInfo(status="FAILED"),
        )

        df = runs_to_dataframe(kfold_runs_with_failure, metric_name="test_roc_auc")

        aggregated = aggregate_kfold_metrics(df)
        filtered = filter_complete_groups(aggregated)

        assert len(filtered) < len(aggregated)

    def test_rank_summary_with_kfold_data(self, mock_fetch_runs: MagicMock):
        """Verify rank summary statistics work with k-fold aggregated data."""
        mock_fetch_runs.return_value = self.kfold_runs

        results = analyze_experiment(
            tracking_uri="http://localhost:5001",
            experiment_id="test_exp",
            metric_name="test_roc_auc",
            aggregate_kfold=True,
        )

        assert _POOLING_DIM in results
        rank_summary = summarize_ranks_by_design_choice(results[_POOLING_DIM])

        assert len(rank_summary) == 2
        assert set(rank_summary["design_choice"].unique()) == {"dmon", "mincut"}
        assert "mean" in rank_summary.columns
        assert "count" in rank_summary.columns
        assert "std" in rank_summary.columns
        assert "median" in rank_summary.columns
