"""Shared pytest fixtures."""

import pytest
import ray

from stgym.rct_utils import DESIGN_DIM_TO_MLFLOW_PATH


@pytest.fixture(scope="session")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


from tests.mock_mlflow import MockRun, MockRunData, MockRunInfo

_POOLING_DIM = "model.pooling.type"
_POOLING_PARAM = DESIGN_DIM_TO_MLFLOW_PATH[_POOLING_DIM]


@pytest.fixture
def regular_runs() -> list[MockRun]:
    """6 mock runs: 3 groups × 2 design choices (dmon, mincut), no k-fold."""
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


@pytest.fixture
def kfold_runs() -> list[MockRun]:
    """12 mock runs: 2 groups × 2 design choices × 3 folds."""
    runs = []
    for group_id in range(2):
        for design_choice in ["dmon", "mincut"]:
            for fold in range(3):
                base_metric = 0.80 + group_id * 0.03
                if design_choice == "dmon":
                    base_metric += 0.02
                metric = base_metric + (fold - 1) * 0.01
                runs.append(
                    MockRun(
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
                )
    return runs


@pytest.fixture
def mixed_runs(regular_runs: list[MockRun], kfold_runs: list[MockRun]) -> list[MockRun]:
    """18 mock runs combining regular and k-fold runs."""
    return regular_runs + kfold_runs
