from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stgym.config_schema import GraphClassifierModelConfig, NodeClassifierModelConfig
from stgym.design_space.schema import TaskReprDesignSpace
from stgym.task_repr import (
    TaskFreeDesign,
    build_performance_matrix,
    expected_mlflow_run_count,
    sample_task_free_designs,
    select_anchor_models,
)
from stgym.utils import load_yaml
from tests.mock_mlflow import MockRun, MockRunData


@pytest.fixture
def node_clf_space() -> TaskReprDesignSpace:
    data = load_yaml("./tests/data/task-repr-design-space-node-clf.yaml")
    return TaskReprDesignSpace.model_validate(data)


@pytest.fixture
def graph_clf_space() -> TaskReprDesignSpace:
    data = load_yaml("./tests/data/task-repr-design-space-graph-clf.yaml")
    return TaskReprDesignSpace.model_validate(data)


class TestSampleTaskFreeDesigns:
    N = 8
    SEED = 42

    @pytest.mark.parametrize(
        "task_type,fixture",
        [
            ("node-classification", "node_clf_space"),
            ("graph-classification", "graph_clf_space"),
        ],
    )
    def test_returns_correct_count(self, task_type, fixture, request):
        space = request.getfixturevalue(fixture)
        designs = sample_task_free_designs(task_type, space, self.N, self.SEED)
        assert len(designs) == self.N

    @pytest.mark.parametrize(
        "task_type,fixture",
        [
            ("node-classification", "node_clf_space"),
            ("graph-classification", "graph_clf_space"),
        ],
    )
    def test_design_ids_are_sequential(self, task_type, fixture, request):
        space = request.getfixturevalue(fixture)
        designs = sample_task_free_designs(task_type, space, self.N, self.SEED)
        assert [d.design_id for d in designs] == list(range(self.N))

    @pytest.mark.parametrize(
        "task_type,fixture",
        [
            ("node-classification", "node_clf_space"),
            ("graph-classification", "graph_clf_space"),
        ],
    )
    def test_reproducible_with_same_seed(self, task_type, fixture, request):
        space = request.getfixturevalue(fixture)
        first = sample_task_free_designs(task_type, space, self.N, self.SEED)
        second = sample_task_free_designs(task_type, space, self.N, self.SEED)
        for a, b in zip(first, second):
            assert a.model == b.model
            assert a.train == b.train
            assert a.data_loader == b.data_loader

    @pytest.mark.parametrize(
        "task_type,fixture",
        [
            ("node-classification", "node_clf_space"),
            ("graph-classification", "graph_clf_space"),
        ],
    )
    def test_different_seeds_produce_different_designs(
        self, task_type, fixture, request
    ):
        space = request.getfixturevalue(fixture)
        first = sample_task_free_designs(task_type, space, self.N, seed=42)
        second = sample_task_free_designs(task_type, space, self.N, seed=99)
        models_equal = all(a.model == b.model for a, b in zip(first, second))
        assert not models_equal

    def test_node_clf_produces_node_classifier_model(self, node_clf_space):
        designs = sample_task_free_designs(
            "node-classification", node_clf_space, self.N, self.SEED
        )
        for d in designs:
            assert isinstance(d.model, NodeClassifierModelConfig)

    def test_graph_clf_produces_graph_classifier_model(self, graph_clf_space):
        designs = sample_task_free_designs(
            "graph-classification", graph_clf_space, self.N, self.SEED
        )
        for d in designs:
            assert isinstance(d.model, GraphClassifierModelConfig)

    def test_node_clf_has_no_pooling(self, node_clf_space):
        designs = sample_task_free_designs(
            "node-classification", node_clf_space, self.N, self.SEED
        )
        for d in designs:
            assert all(mp.pooling is None for mp in d.model.mp_layers)

    def test_returns_task_free_designs(self, node_clf_space):
        designs = sample_task_free_designs(
            "node-classification", node_clf_space, self.N, self.SEED
        )
        for d in designs:
            assert isinstance(d, TaskFreeDesign)
            assert not hasattr(d, "task")


class TestExpectedMlflowRunCount:
    # Tasks from conf/obtain_task_repr_node_clf.yaml
    NODE_CLF_TASKS = [
        "breast-cancer",
        "human-intestine",  # k=8
        "colorectal-cancer",  # k=4
        "upmc",
        "charville",
        "mouse-spleen",
        "human-crc",
        "human-pancreas",  # k=3
        "human-lung",
        "cellcontrast-breast",  # k=5
    ]

    def test_node_clf_smoke_count(self):
        # 4 designs × (1+8+4+1+1+1+1+3+1+5) = 4 × 26 = 104
        assert expected_mlflow_run_count(4, self.NODE_CLF_TASKS) == 104

    def test_scales_linearly_with_n_designs(self):
        count_4 = expected_mlflow_run_count(4, self.NODE_CLF_TASKS)
        count_8 = expected_mlflow_run_count(8, self.NODE_CLF_TASKS)
        assert count_8 == 2 * count_4

    def test_non_kfold_only_tasks(self):
        tasks = ["upmc", "charville", "mouse-spleen"]
        assert expected_mlflow_run_count(3, tasks) == 9  # 3 designs × 3 tasks × 1 run


def _make_run(design_id: int, dataset_name: str, metric: str, value: float) -> MockRun:
    return MockRun(
        data=MockRunData(
            tags={"design_id": str(design_id), "dataset_name": dataset_name},
            metrics={metric: value},
        )
    )


class TestBuildPerformanceMatrix:
    METRIC = "test_accuracy"
    TASK_TYPE = "node-classification"

    def _runs(self, design_id, dataset_name, value):
        return _make_run(design_id, dataset_name, self.METRIC, value)

    def test_basic_shape(self):
        runs = [
            self._runs(0, "ds-a", 0.8),
            self._runs(0, "ds-b", 0.6),
            self._runs(1, "ds-a", 0.7),
            self._runs(1, "ds-b", 0.5),
        ]
        matrix = build_performance_matrix(runs, self.TASK_TYPE)
        assert matrix.shape == (2, 2)
        assert list(matrix.index) == [0, 1]
        assert sorted(matrix.columns) == ["ds-a", "ds-b"]

    def test_values_correct(self):
        runs = [
            self._runs(0, "ds-a", 0.8),
            self._runs(1, "ds-a", 0.6),
        ]
        matrix = build_performance_matrix(runs, self.TASK_TYPE)
        assert matrix.loc[0, "ds-a"] == pytest.approx(0.8)
        assert matrix.loc[1, "ds-a"] == pytest.approx(0.6)

    def test_kfold_runs_averaged(self):
        runs = [
            self._runs(0, "ds-a", 0.6),
            self._runs(0, "ds-a", 0.8),
            self._runs(0, "ds-a", 0.7),
        ]
        matrix = build_performance_matrix(runs, self.TASK_TYPE)
        assert matrix.loc[0, "ds-a"] == pytest.approx(0.7)

    def test_missing_cell_is_nan(self):
        runs = [
            self._runs(0, "ds-a", 0.8),
            self._runs(1, "ds-b", 0.6),
        ]
        matrix = build_performance_matrix(runs, self.TASK_TYPE)
        assert np.isnan(matrix.loc[0, "ds-b"])
        assert np.isnan(matrix.loc[1, "ds-a"])

    def test_uses_correct_metric_for_graph_clf(self):
        runs = [_make_run(0, "ds-a", "test_roc_auc", 0.9)]
        matrix = build_performance_matrix(runs, "graph-classification")
        assert matrix.loc[0, "ds-a"] == pytest.approx(0.9)


class TestSelectAnchorModels:
    def _matrix(self, scores: dict[int, list[float]], datasets=None) -> "pd.DataFrame":
        if datasets is None:
            datasets = [f"ds-{i}" for i in range(len(next(iter(scores.values()))))]
        return pd.DataFrame.from_dict(scores, orient="index", columns=datasets)

    def test_returns_correct_count(self):
        scores = {i: [float(i)] for i in range(12)}
        matrix = self._matrix(scores)
        anchors = select_anchor_models(matrix, n_anchors=4)
        assert len(anchors) == 4

    def test_anchors_span_performance_range(self):
        # 8 designs with scores 0..7, 4 anchors → bins [0,1], [2,3], [4,5], [6,7]
        # upper-mid (n//2=1) of each bin → designs 1, 3, 5, 7
        scores = {i: [float(i)] for i in range(8)}
        matrix = self._matrix(scores)
        anchors = select_anchor_models(matrix, n_anchors=4)
        assert anchors == [1, 3, 5, 7]

    def test_all_nan_rows_dropped(self):
        # designs 1 and 2 have partial NaN → kept (mean computed over valid values)
        # design 4 is all-NaN → excluded
        matrix = pd.DataFrame(
            {
                "ds-a": [0.8, float("nan"), 0.6, 0.4, float("nan")],
                "ds-b": [0.7, 0.9, float("nan"), 0.3, float("nan")],
            },
            index=[0, 1, 2, 3, 4],
        )
        anchors = select_anchor_models(matrix, n_anchors=2)
        assert 4 not in anchors

    def test_warns_when_all_nan_column_dropped(self):
        matrix = pd.DataFrame(
            {
                "ds-a": [0.8, 0.6, 0.4, 0.2],
                "ds-bad": [float("nan")] * 4,
            }
        )
        with patch("stgym.task_repr.logger") as mock_logger:
            select_anchor_models(matrix, n_anchors=2)
        mock_logger.warning.assert_called_once()
        assert "ds-bad" in mock_logger.warning.call_args[0][0]

    def test_raises_if_too_few_complete_designs(self):
        matrix = pd.DataFrame(
            {"ds-a": [float("nan"), float("nan")], "ds-b": [float("nan"), float("nan")]}
        )
        with pytest.raises(ValueError, match="complete designs"):
            select_anchor_models(matrix, n_anchors=2)
