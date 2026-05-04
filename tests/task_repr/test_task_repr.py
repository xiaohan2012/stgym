import pytest

from stgym.config_schema import GraphClassifierModelConfig, NodeClassifierModelConfig
from stgym.design_space.schema import TaskReprDesignSpace
from stgym.task_repr import (
    TaskFreeDesign,
    expected_mlflow_run_count,
    sample_task_free_designs,
)
from stgym.utils import load_yaml


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
