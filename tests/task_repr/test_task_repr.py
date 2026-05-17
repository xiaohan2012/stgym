from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stgym.config_schema import NodeClassifierModelConfig
from stgym.design_space.schema import TaskReprDesignSpace
from stgym.task_repr import (
    build_performance_matrix,
    compute_fingerprints,
    pairwise_similarity,
    sample_designs,
    select_anchor_models,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def node_clf_space():
    return TaskReprDesignSpace.from_yaml(
        "tests/data/task-repr-design-space-node-clf.yaml"
    )


def _make_run(design_id, dataset, metric_value, fold=None, status="FINISHED"):
    tags = {"design_id": str(design_id), "dataset_name": dataset}
    if fold is not None:
        tags["fold"] = str(fold)
    return SimpleNamespace(
        data=SimpleNamespace(
            tags=tags,
            metrics={"test_roc_auc": metric_value},
        ),
        info=SimpleNamespace(status=status),
    )


def _make_perf_matrix(n_designs: int, n_tasks: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.uniform(0, 1, size=(n_designs, n_tasks))
    return pd.DataFrame(
        data,
        index=list(range(n_designs)),
        columns=[f"task_{i}" for i in range(n_tasks)],
    )


def _make_fp_matrix(scores: dict) -> pd.DataFrame:
    """scores: {design_id: {task: value}}"""
    return pd.DataFrame(scores).T


# ---------------------------------------------------------------------------
# sample_designs
# ---------------------------------------------------------------------------


class TestSampleDesigns:
    @property
    def task_type(self) -> str:
        return "node-classification"

    def test_count(self, node_clf_space):
        designs = sample_designs(node_clf_space, self.task_type, n=5, seed=42)
        assert len(designs) == 5

    def test_ids_are_sequential(self, node_clf_space):
        ids = [
            d[0] for d in sample_designs(node_clf_space, self.task_type, n=4, seed=42)
        ]
        assert ids == list(range(4))

    def test_no_task_in_partial(self, node_clf_space):
        for _, partial in sample_designs(node_clf_space, self.task_type, n=3, seed=42):
            assert "task" not in partial
            assert set(partial.keys()) == {"model", "train", "data_loader"}

    def test_model_type_matches_task_type(self, node_clf_space):
        for _, partial in sample_designs(node_clf_space, self.task_type, n=3, seed=42):
            assert isinstance(partial["model"], NodeClassifierModelConfig)

    def test_reproducible_with_same_seed(self, node_clf_space):
        a = sample_designs(node_clf_space, self.task_type, n=5, seed=7)
        b = sample_designs(node_clf_space, self.task_type, n=5, seed=7)
        for (id_a, pa), (id_b, pb) in zip(a, b):
            assert id_a == id_b
            assert pa["model"].model_dump() == pb["model"].model_dump()
            assert pa["train"].model_dump() == pb["train"].model_dump()
            assert pa["data_loader"].model_dump() == pb["data_loader"].model_dump()

    def test_different_seeds_produce_different_configs(self, node_clf_space):
        a = [
            pa["model"].model_dump()
            for _, pa in sample_designs(node_clf_space, self.task_type, n=5, seed=1)
        ]
        b = [
            pb["model"].model_dump()
            for _, pb in sample_designs(node_clf_space, self.task_type, n=5, seed=2)
        ]
        assert a != b


# ---------------------------------------------------------------------------
# build_performance_matrix
# ---------------------------------------------------------------------------


class TestBuildPerformanceMatrix:
    @property
    def simple_runs(self):
        return [
            _make_run(0, "brca", 0.8),
            _make_run(0, "glioblastoma", 0.7),
            _make_run(1, "brca", 0.6),
            _make_run(1, "glioblastoma", 0.9),
        ]

    def test_shape(self):
        assert build_performance_matrix(self.simple_runs, "test_roc_auc").shape == (
            2,
            2,
        )

    def test_index_and_columns(self):
        m = build_performance_matrix(self.simple_runs, "test_roc_auc")
        assert set(m.index) == {0, 1}
        assert set(m.columns) == {"brca", "glioblastoma"}

    def test_values(self):
        m = build_performance_matrix(self.simple_runs, "test_roc_auc")
        assert m.loc[0, "brca"] == pytest.approx(0.8)
        assert m.loc[1, "glioblastoma"] == pytest.approx(0.9)

    def test_failed_runs_excluded(self):
        runs = self.simple_runs + [_make_run(2, "brca", 0.5, status="FAILED")]
        assert 2 not in build_performance_matrix(runs, "test_roc_auc").index

    def test_kfold_aggregated(self):
        runs = [
            _make_run(0, "human-intestine", 0.6, fold=0),
            _make_run(0, "human-intestine", 0.8, fold=1),
            _make_run(0, "brca", 0.7),
        ]
        m = build_performance_matrix(runs, "test_roc_auc", min_task_coverage=0.0)
        assert m.loc[0, "human-intestine"] == pytest.approx(0.7)

    def test_min_task_coverage_filters_sparse_designs(self):
        runs = self.simple_runs + [_make_run(2, "brca", 0.5)]
        assert (
            2
            not in build_performance_matrix(
                runs, "test_roc_auc", min_task_coverage=1.0
            ).index
        )

    def test_empty_runs_returns_empty_df(self):
        assert build_performance_matrix([], "test_roc_auc").empty

    def test_runs_missing_design_id_skipped(self):
        bad = SimpleNamespace(
            data=SimpleNamespace(
                tags={"dataset_name": "brca"},
                metrics={"test_roc_auc": 0.5},
            ),
            info=SimpleNamespace(status="FINISHED"),
        )
        assert build_performance_matrix([bad], "test_roc_auc").empty


# ---------------------------------------------------------------------------
# select_anchor_models
# ---------------------------------------------------------------------------


class TestSelectAnchorModels:
    @property
    def matrix(self) -> pd.DataFrame:
        return _make_perf_matrix(n_designs=48, n_tasks=5, seed=42)

    def test_returns_correct_count(self):
        assert len(select_anchor_models(self.matrix, n_anchors=12)) == 12

    def test_returns_valid_design_ids(self):
        for a in select_anchor_models(self.matrix, n_anchors=12):
            assert a in self.matrix.index

    def test_no_duplicate_anchors(self):
        anchors = select_anchor_models(self.matrix, n_anchors=12)
        assert len(set(anchors)) == len(anchors)

    def test_anchors_span_performance_range(self):
        anchors = select_anchor_models(self.matrix, n_anchors=4)
        avg = self.matrix.mean(axis=1)
        scores = avg.loc[anchors].sort_values().values
        overall_median = avg.median()
        assert scores[0] < overall_median
        assert scores[-1] > overall_median

    @pytest.mark.parametrize("n_anchors", [1, 4, 12])
    def test_various_anchor_counts(self, n_anchors):
        assert len(select_anchor_models(self.matrix, n_anchors=n_anchors)) == n_anchors

    def test_raises_when_too_few_designs(self):
        with pytest.raises(ValueError, match="Not enough designs"):
            select_anchor_models(
                _make_perf_matrix(n_designs=3, n_tasks=2), n_anchors=12
            )

    def test_deterministic(self):
        assert select_anchor_models(self.matrix, n_anchors=6) == select_anchor_models(
            self.matrix, n_anchors=6
        )


# ---------------------------------------------------------------------------
# compute_fingerprints
# ---------------------------------------------------------------------------


class TestComputeFingerprints:
    @property
    def matrix(self) -> pd.DataFrame:
        return _make_fp_matrix(
            {
                0: {"task_a": 0.9, "task_b": 0.5},
                1: {"task_a": 0.4, "task_b": 0.8},
                2: {"task_a": 0.6, "task_b": 0.6},
            }
        )

    def test_shape(self):
        assert compute_fingerprints(self.matrix, anchor_ids=[0, 1]).shape == (2, 2)

    def test_index_is_tasks(self):
        assert set(compute_fingerprints(self.matrix, anchor_ids=[0, 1]).index) == {
            "task_a",
            "task_b",
        }

    def test_columns_are_anchor_ids(self):
        assert set(compute_fingerprints(self.matrix, anchor_ids=[0, 2]).columns) == {
            0,
            2,
        }

    def test_values_correct(self):
        fps = compute_fingerprints(self.matrix, anchor_ids=[0])
        assert fps.loc["task_a", 0] == pytest.approx(0.9)
        assert fps.loc["task_b", 0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# pairwise_similarity
# ---------------------------------------------------------------------------


class TestPairwiseSimilarity:
    @property
    def three_task_fps(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"a0": [0.8, 0.3, 0.6], "a1": [0.2, 0.9, 0.4]},
            index=["t1", "t2", "t3"],
        )

    @property
    def identical_fps(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"a0": [0.9, 0.9], "a1": [0.4, 0.4], "a2": [0.6, 0.6]},
            index=["task_a", "task_b"],
        )

    @property
    def opposite_fps(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"a0": [0.1, 0.9], "a1": [0.5, 0.5], "a2": [0.9, 0.1]},
            index=["task_a", "task_b"],
        )

    def test_diagonal_is_one(self):
        sim = pairwise_similarity(self.three_task_fps)
        for task in self.three_task_fps.index:
            assert sim.loc[task, task] == pytest.approx(1.0)

    def test_symmetric(self):
        sim = pairwise_similarity(self.three_task_fps)
        np.testing.assert_allclose(sim.values, sim.values.T)

    def test_identical_rankings_give_one(self):
        assert pairwise_similarity(self.identical_fps).loc[
            "task_a", "task_b"
        ] == pytest.approx(1.0)

    def test_opposite_rankings_give_minus_one(self):
        assert pairwise_similarity(self.opposite_fps).loc[
            "task_a", "task_b"
        ] == pytest.approx(-1.0)

    def test_output_shape(self):
        assert pairwise_similarity(self.three_task_fps).shape == (3, 3)

    def test_index_and_columns_match_tasks(self):
        fps = pd.DataFrame({"a0": [0.5, 0.6]}, index=["brca", "glioblastoma"])
        sim = pairwise_similarity(fps)
        assert list(sim.index) == ["brca", "glioblastoma"]
        assert list(sim.columns) == ["brca", "glioblastoma"]
