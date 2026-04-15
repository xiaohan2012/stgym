"""Tests for stgym/rct_utils.py."""

from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stgym.rct_utils import (
    DESIGN_DIM_TO_MLFLOW_PATH,
    aggregate_kfold_metrics,
    compute_within_group_ranks,
    filter_complete_groups,
    runs_to_dataframe,
    summarize_ranks_by_design_choice,
)
from tests.mock_mlflow import MockRun, MockRunData, MockRunInfo

_DESIGN_DIM = "model.use_batchnorm"
_MLFLOW_PARAM = DESIGN_DIM_TO_MLFLOW_PATH[_DESIGN_DIM]


class TestRunsToDataframe:
    """Test runs_to_dataframe conversion."""

    @property
    def mock_runs(self) -> list[MockRun]:
        """Create mock MLflow Run objects with design_dimension tag."""
        runs = []
        for i in range(4):
            run = MockRun(
                data=MockRunData(
                    tags={
                        "group_id": str(i // 2),
                        "fold": str(i % 2),
                        "design_dimension": _DESIGN_DIM,
                    },
                    metrics={"test_roc_auc": 0.8 + i * 0.05},
                    params={_MLFLOW_PARAM: str(i % 2 == 0).lower()},
                ),
                info=MockRunInfo(status="FINISHED"),
            )
            runs.append(run)
        return runs

    def test_extracts_correct_columns(self):
        """Verify all expected columns are extracted."""
        df = runs_to_dataframe(self.mock_runs, metric_name="test_roc_auc")

        assert set(df.columns) == {
            "group_id",
            "fold",
            "metric",
            "design_choice",
            "design_dimension",
            "run_status",
        }
        assert len(df) == 4

    def test_design_dimension_from_tag(self):
        """Verify design_dimension column is populated from run tags."""
        df = runs_to_dataframe(self.mock_runs, metric_name="test_roc_auc")

        assert (df["design_dimension"] == _DESIGN_DIM).all()

    def test_design_choice_resolved_via_mapping(self):
        """Verify design_choice is resolved from DESIGN_DIM_TO_MLFLOW_PATH."""
        df = runs_to_dataframe(self.mock_runs, metric_name="test_roc_auc")

        assert df["design_choice"].notna().all()

    def test_handles_missing_values(self):
        """Verify graceful handling when run lacks expected fields."""
        run = MockRun(
            data=MockRunData(tags={}, metrics={}, params={}),
            info=MockRunInfo(status="FAILED"),
        )

        df = runs_to_dataframe([run], metric_name="test_roc_auc")

        assert len(df) == 1
        assert df["group_id"].iloc[0] is None
        assert df["metric"].iloc[0] is None
        assert df["design_dimension"].iloc[0] is None
        assert df["design_choice"].iloc[0] is None


class TestAggregateKfoldMetrics:
    """Test k-fold aggregation logic."""

    @property
    def kfold_df(self) -> pd.DataFrame:
        """Sample DataFrame with k-fold runs."""
        return pd.DataFrame(
            {
                "group_id": ["0", "0", "0", "0", "1", "1", "1", "1"],
                "design_choice": ["a", "a", "b", "b", "a", "a", "b", "b"],
                "fold": ["0", "1", "0", "1", "0", "1", "0", "1"],
                "metric": [0.8, 0.9, 0.7, 0.75, 0.85, 0.95, 0.72, 0.78],
                "run_status": ["FINISHED"] * 8,
            }
        )

    @property
    def non_kfold_df(self) -> pd.DataFrame:
        """Sample DataFrame without k-fold runs."""
        return pd.DataFrame(
            {
                "group_id": ["0", "0", "1", "1"],
                "design_choice": ["a", "b", "a", "b"],
                "fold": [None, None, None, None],
                "metric": [0.85, 0.725, 0.9, 0.75],
                "run_status": ["FINISHED"] * 4,
            }
        )

    def test_aggregates_kfold_by_mean(self):
        """Verify k-fold metrics are averaged correctly."""
        result = aggregate_kfold_metrics(self.kfold_df)

        # Should have 4 rows: 2 groups x 2 design choices
        assert len(result) == 4

        # Check aggregated values for group 0, design a
        row = result[(result["group_id"] == "0") & (result["design_choice"] == "a")]
        assert row["metric"].iloc[0] == pytest.approx(0.85)  # mean of 0.8, 0.9

    def test_non_kfold_passes_through(self):
        """Verify non-k-fold runs are not modified."""
        result = aggregate_kfold_metrics(self.non_kfold_df)

        assert len(result) == len(self.non_kfold_df)
        pd.testing.assert_frame_equal(result, self.non_kfold_df)

    def test_failed_fold_marks_group_failed(self):
        """Verify a single failed fold marks the aggregated run as failed."""
        df = self.kfold_df.copy()
        df.loc[0, "run_status"] = "FAILED"

        result = aggregate_kfold_metrics(df)

        # Group 0, design a should be marked FAILED
        row = result[(result["group_id"] == "0") & (result["design_choice"] == "a")]
        assert row["run_status"].iloc[0] == "FAILED"


class TestFilterCompleteGroups:
    """Test filtering of incomplete groups."""

    @property
    def mixed_df(self) -> pd.DataFrame:
        """DataFrame with complete and incomplete groups."""
        return pd.DataFrame(
            {
                "group_id": ["0", "0", "1", "1", "2", "2"],
                "design_choice": ["a", "b", "a", "b", "a", "b"],
                "metric": [0.8, 0.9, 0.7, 0.8, 0.85, 0.95],
                "run_status": [
                    "FINISHED",
                    "FINISHED",
                    "FAILED",
                    "FINISHED",
                    "FINISHED",
                    "FINISHED",
                ],
            }
        )

    def test_removes_incomplete_groups(self):
        """Verify groups with any failed run are removed."""
        result = filter_complete_groups(self.mixed_df)

        # Group 1 should be removed (has a failed run)
        assert set(result["group_id"]) == {"0", "2"}
        assert len(result) == 4

    def test_all_complete_unchanged(self):
        """Verify all-complete DataFrame is unchanged."""
        complete_df = pd.DataFrame(
            {
                "group_id": ["0", "0"],
                "run_status": ["FINISHED", "FINISHED"],
            }
        )
        result = filter_complete_groups(complete_df)

        assert len(result) == 2


class TestComputeWithinGroupRanks:
    """Test within-group rank computation."""

    @property
    def sample_df(self) -> pd.DataFrame:
        """Sample DataFrame for ranking tests."""
        return pd.DataFrame(
            {
                "group_id": ["0", "0", "0", "1", "1", "1"],
                "design_choice": ["a", "b", "c", "a", "b", "c"],
                "metric": [0.9, 0.8, 0.7, 0.6, 0.8, 0.7],
            }
        )

    def test_higher_is_better_by_default(self):
        """Verify higher metric gets rank 1 by default."""
        result = compute_within_group_ranks(self.sample_df)

        # Group 0: a (0.9) should be rank 1, b (0.8) rank 2, c (0.7) rank 3
        group0 = result[result["group_id"] == "0"]
        assert group0[group0["design_choice"] == "a"]["rank"].iloc[0] == 1.0
        assert group0[group0["design_choice"] == "b"]["rank"].iloc[0] == 2.0
        assert group0[group0["design_choice"] == "c"]["rank"].iloc[0] == 3.0

    def test_ascending_mode(self):
        """Verify ascending=True gives lower metric better rank."""
        result = compute_within_group_ranks(self.sample_df, ascending=True)

        # Group 0: c (0.7) should be rank 1 (lowest is best)
        group0 = result[result["group_id"] == "0"]
        assert group0[group0["design_choice"] == "c"]["rank"].iloc[0] == 1.0

    def test_does_not_modify_original(self):
        """Verify original DataFrame is not modified."""
        original = self.sample_df.copy()
        compute_within_group_ranks(self.sample_df)

        pd.testing.assert_frame_equal(self.sample_df, original)


class TestSummarizeRanksByDesignChoice:
    """Test rank summary statistics."""

    @property
    def ranked_df(self) -> pd.DataFrame:
        """DataFrame with precomputed ranks."""
        return pd.DataFrame(
            {
                "design_choice": ["a", "b", "a", "b", "a", "b"],
                "rank": [1.0, 2.0, 2.0, 1.0, 1.0, 2.0],
            }
        )

    def test_computes_correct_statistics(self):
        """Verify mean, count, std, median are computed."""
        result = summarize_ranks_by_design_choice(self.ranked_df)

        assert set(result.columns) == {
            "design_choice",
            "mean",
            "count",
            "std",
            "median",
        }

        # Design a: ranks [1, 2, 1] -> mean ~1.33
        row_a = result[result["design_choice"] == "a"]
        assert row_a["mean"].iloc[0] == pytest.approx(4 / 3)
        assert row_a["count"].iloc[0] == 3
        assert row_a["median"].iloc[0] == 1.0


@mock.patch("stgym.rct_utils.fetch_runs")
class TestAnalyzeExperiment:
    """Test the high-level analyze_experiment function."""

    def test_returns_empty_dict_when_no_runs(self, mock_fetch_runs: MagicMock):
        """Verify empty dict is returned when no runs exist."""
        from stgym.rct_utils import analyze_experiment

        mock_fetch_runs.return_value = []

        result = analyze_experiment(
            tracking_uri="http://localhost:5001",
            experiment_id="123",
        )

        assert result == {}

    def test_full_pipeline_single_dimension(self, mock_fetch_runs: MagicMock):
        """Verify full analysis pipeline runs correctly for a single design dimension."""
        from stgym.rct_utils import analyze_experiment

        runs = []
        for i in range(4):
            run = MockRun(
                data=MockRunData(
                    tags={"group_id": str(i // 2), "design_dimension": _DESIGN_DIM},
                    metrics={"test_roc_auc": 0.8 + i * 0.05},
                    params={_MLFLOW_PARAM: str(i % 2 == 0).lower()},
                ),
                info=MockRunInfo(status="FINISHED"),
            )
            runs.append(run)

        mock_fetch_runs.return_value = runs

        result = analyze_experiment(
            tracking_uri="http://localhost:5001",
            experiment_id="123",
        )

        assert isinstance(result, dict)
        assert _DESIGN_DIM in result
        df = result[_DESIGN_DIM]
        assert isinstance(df, pd.DataFrame)
        assert "rank" in df.columns
        assert len(df) > 0

    def test_multiple_dimensions_analyzed_separately(self, mock_fetch_runs: MagicMock):
        """Verify runs with different design dimensions are analyzed independently."""
        from stgym.rct_utils import analyze_experiment

        dim_a = "model.act"
        param_a = DESIGN_DIM_TO_MLFLOW_PATH[dim_a]
        dim_b = "train.optim.optimizer"
        param_b = DESIGN_DIM_TO_MLFLOW_PATH[dim_b]

        runs = []
        for i in range(4):
            dim = dim_a if i < 2 else dim_b
            param = param_a if i < 2 else param_b
            runs.append(
                MockRun(
                    data=MockRunData(
                        tags={"group_id": "0", "design_dimension": dim},
                        metrics={"test_roc_auc": 0.8 + i * 0.05},
                        params={param: f"choice_{i}"},
                    ),
                    info=MockRunInfo(status="FINISHED"),
                )
            )

        mock_fetch_runs.return_value = runs

        result = analyze_experiment(
            tracking_uri="http://localhost:5001",
            experiment_id="123",
        )

        assert set(result.keys()) == {dim_a, dim_b}
        assert len(result[dim_a]) == 2
        assert len(result[dim_b]) == 2
