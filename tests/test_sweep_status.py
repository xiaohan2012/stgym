"""Unit tests for compute_overall_progress in scripts/sweep_status.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.mock_mlflow import MockRun, MockRunData, MockRunInfo

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from sweep_status import (  # noqa: E402
    _get_k_for_run,
    compute_exp_stats,
    compute_overall_progress,
)


def _mock_run(
    status: str,
    start_time: int = 0,
    end_time: int | None = None,
    dataset_name: str | None = None,
) -> MockRun:
    tags = {"dataset_name": dataset_name} if dataset_name is not None else {}
    return MockRun(
        data=MockRunData(tags=tags),
        info=MockRunInfo(status=status, start_time=start_time, end_time=end_time),
    )


class TestGetKForRun:
    """Tests for _get_k_for_run helper."""

    @pytest.mark.parametrize(
        "dataset_name, expected_k",
        [
            ("human-intestine", 8),
            ("spatial-vdj", 5),
            ("human-pancreas", 3),
            ("colorectal-cancer", 4),
            ("gastric-bladder-cancer", 3),
            ("cellcontrast-breast", 5),
        ],
    )
    def test_kfold_dataset_returns_num_folds(
        self, dataset_name: str, expected_k: int
    ) -> None:
        run = _mock_run("FINISHED", dataset_name=dataset_name)
        assert _get_k_for_run(run) == expected_k

    def test_non_kfold_dataset_returns_one(self) -> None:
        run = _mock_run("FINISHED", dataset_name="brca")
        assert _get_k_for_run(run) == 1

    def test_missing_dataset_tag_returns_one(self) -> None:
        run = _mock_run("FINISHED")
        assert _get_k_for_run(run) == 1


class TestComputeExpStats:
    """Tests for compute_exp_stats."""

    def test_empty_runs_returns_empty(self) -> None:
        assert compute_exp_stats([], now_ms=10_000) == {}

    def test_non_kfold_runs_count_one_per_trial(self) -> None:
        runs = [
            _mock_run("FINISHED", start_time=0, end_time=1_000, dataset_name="brca"),
            _mock_run("FINISHED", start_time=0, end_time=1_000, dataset_name="brca"),
            _mock_run("FAILED", start_time=0, end_time=1_000, dataset_name="brca"),
        ]
        # Total duration: 1 hour
        stats = compute_exp_stats(runs, now_ms=3_600_000)

        assert stats["total_finished"] == 2
        assert stats["total_failed"] == 1
        assert stats["total_completed_trials"] == pytest.approx(3.0)
        assert stats["throughput_trials_per_h"] == pytest.approx(3.0)

    def test_kfold_runs_contribute_fractionally(self) -> None:
        # 5-fold dataset: 5 finished runs == 1 trial
        runs = [
            _mock_run(
                "FINISHED",
                start_time=0,
                end_time=1_000,
                dataset_name="spatial-vdj",
            )
            for _ in range(5)
        ]
        stats = compute_exp_stats(runs, now_ms=3_600_000)

        assert stats["total_finished"] == 5
        assert stats["total_completed_trials"] == pytest.approx(1.0)
        assert stats["throughput_trials_per_h"] == pytest.approx(1.0)

    def test_mixed_kfold_and_non_kfold(self) -> None:
        # 5 fold runs (spatial-vdj, k=5) + 2 non-fold runs (brca, k=1)
        # = 1 trial + 2 trials = 3 completed trials
        runs = [
            _mock_run(
                "FINISHED",
                start_time=0,
                end_time=1_000,
                dataset_name="spatial-vdj",
            )
            for _ in range(5)
        ] + [
            _mock_run("FINISHED", start_time=0, end_time=1_000, dataset_name="brca"),
            _mock_run("FAILED", start_time=0, end_time=1_000, dataset_name="brca"),
        ]
        stats = compute_exp_stats(runs, now_ms=3_600_000)

        assert stats["total_finished"] == 6
        assert stats["total_failed"] == 1
        assert stats["total_completed_trials"] == pytest.approx(3.0)


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
            "total_completed_trials": 42.0,
            "throughput_trials_per_h": 12.0,
        }

    def test_normal_case(self) -> None:
        result = compute_overall_progress(self.base_df, self.base_exp_stats)

        assert result["total_completed_trials"] == pytest.approx(42.0)
        assert result["total_finished"] == 38
        assert result["total_failed"] == 4
        assert result["total_active"] == 3
        assert result["total_stale"] == 1
        assert result["total_expected"] == 100
        assert result["throughput_trials_per_min"] == pytest.approx(0.2)
        # remaining = 100 - 42 = 58, throughput = 0.2 trials/min → eta = 290 min
        assert result["eta_min"] == pytest.approx(290.0)

    def test_zero_throughput_gives_no_eta(self) -> None:
        exp_stats = {**self.base_exp_stats, "throughput_trials_per_h": 0.0}
        result = compute_overall_progress(self.base_df, exp_stats)

        assert result["eta_min"] is None
        assert result["throughput_trials_per_min"] == pytest.approx(0.0)

    def test_all_trials_completed_gives_no_eta(self) -> None:
        exp_stats = {**self.base_exp_stats, "total_completed_trials": 100.0}
        result = compute_overall_progress(self.base_df, exp_stats)

        assert result["total_completed_trials"] == pytest.approx(100.0)
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

    def test_pending_dim_with_overrun_kfold_still_yields_eta(self) -> None:
        # Reproduces the bug from issue #186:
        # k-fold dim has more runs than trials, but a PENDING dim has 0 runs.
        # In trial units, total_completed_trials < total_expected, so ETA is computed.
        df = pd.DataFrame(
            {
                "active": [0, 0],
                "stale": [0, 0],
                "trials": [50.0, 50.0],
                "state": ["STARTED", "PENDING"],
            }
        )
        # STARTED dim has 50 trials worth of finished k-fold runs;
        # PENDING dim contributes 0 completed trials.
        exp_stats = {
            "total_finished": 250,  # 50 trials × 5 folds
            "total_failed": 0,
            "total_completed_trials": 50.0,
            "throughput_trials_per_h": 6.0,
        }
        result = compute_overall_progress(df, exp_stats)

        assert result["total_expected"] == 100
        assert result["total_completed_trials"] == pytest.approx(50.0)
        # Percentage stays at 50%, not over 100% as in the bug.
        assert 100 * result["total_completed_trials"] / result[
            "total_expected"
        ] == pytest.approx(50.0)
        # ETA is computed (not None): remaining = 50, throughput = 0.1 trials/min
        assert result["eta_min"] == pytest.approx(500.0)
