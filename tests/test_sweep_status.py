"""Unit tests for compute_overall_progress in scripts/sweep_status.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from sweep_status import compute_overall_progress  # noqa: E402


class TestComputeOverallProgress:
    """Tests for compute_overall_progress."""

    @property
    def base_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "active": [2, 1],
                "stale": [0, 1],
                "trials": [50.0, 50.0],
                "state": ["STARTED", "STARTED"],
            }
        )

    @property
    def base_exp_stats(self) -> dict:
        return {
            "total_finished": 38,
            "total_failed": 4,
            "throughput_per_h": 12.0,
        }

    def test_normal_case(self) -> None:
        result = compute_overall_progress(self.base_df, self.base_exp_stats)

        assert result["total_completed"] == 42
        assert result["total_finished"] == 38
        assert result["total_failed"] == 4
        assert result["total_active"] == 3
        assert result["total_stale"] == 1
        assert result["total_expected"] == 100
        assert result["throughput_per_min"] == pytest.approx(0.2)
        # remaining = 100 - 42 = 58, throughput = 0.2 runs/min → eta = 290 min
        assert result["eta_min"] == pytest.approx(290.0)

    def test_zero_throughput_gives_no_eta(self) -> None:
        exp_stats = {**self.base_exp_stats, "throughput_per_h": 0.0}
        result = compute_overall_progress(self.base_df, exp_stats)

        assert result["eta_min"] is None
        assert result["throughput_per_min"] == pytest.approx(0.0)

    def test_all_runs_completed_gives_no_eta(self) -> None:
        exp_stats = {**self.base_exp_stats, "total_finished": 100, "total_failed": 0}
        result = compute_overall_progress(self.base_df, exp_stats)

        assert result["total_completed"] == 100
        assert result["total_expected"] == 100
        assert result["eta_min"] is None

    def test_all_trials_null_gives_no_expected_and_no_eta(self) -> None:
        df = self.base_df.copy()
        df["trials"] = None
        result = compute_overall_progress(df, self.base_exp_stats)

        assert result["total_expected"] is None
        assert result["eta_min"] is None

    @pytest.mark.parametrize(
        "trials, expected_total",
        [
            ([50.0, None], 50),
            ([None, 75.0], 75),
            ([30.0, 70.0], 100),
        ],
    )
    def test_mixed_null_trials_sums_non_null(
        self, trials: list, expected_total: int
    ) -> None:
        df = self.base_df.copy()
        df["trials"] = trials
        result = compute_overall_progress(df, self.base_exp_stats)

        assert result["total_expected"] == expected_total
