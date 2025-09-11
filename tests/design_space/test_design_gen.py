import pydash as _
import pytest

from stgym.config_schema import (
    ClusteringModelConfig,
    DataLoaderConfig,
    ExperimentConfig,
    GraphClassifierModelConfig,
    NodeClassifierModelConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.design_space.design_gen import (
    generate_data_loader_config,
    generate_experiment,
    generate_model_config,
    generate_task_config,
    generate_train_config,
    sample_across_dimensions,
)
from stgym.design_space.schema import DataLoaderSpace, DesignSpace, ModelSpace
from stgym.utils import load_yaml

from ..utils import RANDOM_SEEDS


@pytest.fixture
def mock_node_clf_design_space():
    data = load_yaml("./tests/data/design-space-node-clf.yaml")

    return DesignSpace.model_validate(data)


@pytest.fixture
def mock_graph_clf_design_space():
    data = load_yaml("./tests/data/design-space-graph-clf.yaml")

    return DesignSpace.model_validate(data)


@pytest.fixture
def mock_clustering_design_space():
    data = load_yaml("./tests/data/design-space-clustering.yaml")

    return DesignSpace.model_validate(data)


# A factory fixture that yields the other data sources
@pytest.fixture(
    params=[
        "mock_node_clf_design_space",
        "mock_graph_clf_design_space",
        "mock_clustering_design_space",
    ]
)
def mock_design_space(request):
    # 'request.param' will be the string name of the fixture
    # This dynamically requests the fixture by name
    return request.getfixturevalue(request.param)


class TestSampleAcrossDimensions:
    @property
    def space(self):
        return ModelSpace(
            num_mp_layers=1,
            global_pooling="mean",
            normalize_adj=True,
            layer_type="ginconv",
            dim_inner=[64, 128],
            act=["prelu", "relu"],
            use_batchnorm=True,
            pooling=dict(type="dmon", n_clusters=[10, 20]),
            post_mp_dims=["64,32", "32, 16"],
        )

    def test_basic(self):

        design = sample_across_dimensions(self.space)
        assert design["num_mp_layers"] == 1

        assert design["normalize_adj"] is True
        assert design["pooling"]["type"] == "dmon"
        assert design["pooling"]["n_clusters"] in (10, 20)
        assert design["layer_type"] == "ginconv"

        assert design["global_pooling"] == "mean"

        assert design["post_mp_dims"] in ["64,32", "32, 16"]

    def test_with_none_values(self):
        space = self.space
        space.global_pooling = None
        space.post_mp_dims = None
        design = sample_across_dimensions(space)
        assert design["global_pooling"] == None
        assert design["post_mp_dims"] == None

    @pytest.mark.parametrize("seed", RANDOM_SEEDS)
    def test_seed(self, seed):
        design = sample_across_dimensions(self.space, seed=seed)
        same_design = sample_across_dimensions(self.space, seed=seed)
        assert design == same_design


class TestGenerateDataLoaderConfig:
    knn_k_choices = (1, 2)
    radius_ratio_choices = (0.1, 0.2)

    @property
    def space(self):
        return DataLoaderSpace(
            graph_const=["knn", "radius"],
            knn_k=self.knn_k_choices,
            radius_ratio=self.radius_ratio_choices,
            batch_size=[8, 16],
        )

    @pytest.mark.parametrize("seed", RANDOM_SEEDS)
    def test_basic(self, seed: int):
        """That knn_k/radius_ratio params are conditional on the choice of graph_const."""
        design = generate_data_loader_config(self.space, k=1, seed=seed)[0]
        if design.graph_const == "knn":
            assert design.radius_ratio is None
            assert design.knn_k in self.knn_k_choices
        else:
            assert design.knn_k is None
            assert design.radius_ratio in self.radius_ratio_choices

    def test_both_graph_const_are_covered(self):
        designs = generate_data_loader_config(self.space, k=10, seed=42)
        assert len(_.uniq(_.map_(designs, "graph_const"))) == 2


class TestGenerateDesign:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_multiplicty(self, k, mock_design_space):

        experiments = generate_experiment(mock_design_space, k=k)
        assert len(experiments) == k
        for exp in experiments:
            assert isinstance(exp, ExperimentConfig)

    def test_task_config_validity(self, mock_design_space):
        configs = generate_task_config(mock_design_space.task, k=100)
        for config in configs:
            assert isinstance(config, TaskConfig)

            assert config.type == mock_design_space.task.type
            if config.type == "node-classification":
                assert config.num_classes == 10
                assert config.dataset_name == "human-crc"
            if config.type == "graph-classification":
                assert config.num_classes == 2
                assert config.dataset_name == "brca"

    @pytest.mark.parametrize("seed", RANDOM_SEEDS)
    def test_consistency_under_fixed_random_seed(self, mock_design_space, seed):
        exps1 = generate_experiment(mock_design_space, k=5, seed=seed)
        exps2 = generate_experiment(mock_design_space, k=5, seed=seed)
        assert exps1 == exps2

    @pytest.mark.parametrize("seed", RANDOM_SEEDS)
    def test_inner_randomness(self, mock_design_space, seed):
        exp1, exp2 = generate_experiment(mock_design_space, k=2, seed=seed)
        assert exp1 != exp2


class TestModelConfig:
    def test_graph_clf_model_config(self, mock_graph_clf_design_space):
        config = generate_model_config(
            "graph-classification", mock_graph_clf_design_space.model, k=1
        )[0]
        assert isinstance(config, GraphClassifierModelConfig)
        assert config.post_mp_layer.dims in ([64, 32], [32, 16])
        assert config.mp_layers[0].use_batchnorm is True
        assert config.post_mp_layer.use_batchnorm is True

    def test_node_clf_model_config(self, mock_node_clf_design_space):
        config = generate_model_config(
            "node-classification", mock_node_clf_design_space.model, k=1
        )[0]
        assert isinstance(config, NodeClassifierModelConfig)
        assert config.post_mp_layer.dims in ([64, 32], [32, 16])
        assert config.mp_layers[0].use_batchnorm is True
        assert config.post_mp_layer.use_batchnorm is True

    def test_clustering_model_config(self, mock_clustering_design_space):
        config = generate_model_config(
            "node-clustering", mock_clustering_design_space.model, k=1
        )[0]
        assert isinstance(config, ClusteringModelConfig)
        assert config.mp_layers[0].use_batchnorm is True


def test_train_config_validity(mock_design_space):
    config = generate_train_config(mock_design_space.train, k=1)[0]
    assert isinstance(config, TrainConfig)
    assert config.optim.optimizer == "adam"
    assert config.max_epoch in (10, 100)


def test_data_loader_config_validity(mock_design_space):
    config = generate_data_loader_config(mock_design_space.data_loader, k=1)[0]
    assert isinstance(config, DataLoaderConfig)
    assert config.graph_const in ["knn", "radius"]
