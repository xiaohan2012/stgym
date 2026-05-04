import pytest

from stgym.config_schema import GraphClassifierModelConfig, NodeClassifierModelConfig
from stgym.design_space.schema import TaskReprDesignSpace
from stgym.task_repr import TaskFreeDesign, sample_task_free_designs
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
