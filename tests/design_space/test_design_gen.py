import pytest

from stgym.config_schema import (
    DataLoaderConfig,
    ExperimentConfig,
    GraphClassifierModelConfig,
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
from stgym.design_space.schema import DesignSpace, ModelSpace, TaskSpace
from stgym.utils import load_yaml

from ..utils import RANDOM_SEEDS


@pytest.fixture
def mock_design_space():
    data = load_yaml("./tests/data/design-space-example.yaml")

    return DesignSpace.model_validate(data)


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

    @property
    def space_with_zip(self):
        return TaskSpace(
            zip_=["dataset_name", "type"],
            dataset_name=["a", "b"],
            type=["graph-classification", "node-classification"],
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

    @pytest.mark.parametrize("seed", RANDOM_SEEDS)
    def test_with_zip(self, seed):
        design = sample_across_dimensions(self.space_with_zip, seed=seed)
        if design["dataset_name"] == "a":
            assert design["type"] == "graph-classification"
        else:
            assert design["type"] == "node-classification"


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
            assert config.dataset_name in ["brca", "animal"]
            assert config.type in ["graph-classification", "node-clustering"]

            # ensure that the two fields are zipped
            if config.dataset_name == "brca":
                assert config.type == "graph-classification"
            else:
                assert config.type == "node-clustering"

    @pytest.mark.parametrize("seed", RANDOM_SEEDS)
    def test_consistency_under_fixed_random_seed(self, mock_design_space, seed):
        exps1 = generate_experiment(mock_design_space, k=5, seed=seed)
        exps2 = generate_experiment(mock_design_space, k=5, seed=seed)
        assert exps1 == exps2

    @pytest.mark.parametrize("seed", RANDOM_SEEDS)
    def test_inner_randomness(self, mock_design_space, seed):
        exp1, exp2 = generate_experiment(mock_design_space, k=2, seed=seed)
        assert exp1 != exp2


def test_model_config_validity(mock_design_space):
    config = generate_model_config(mock_design_space.model, k=1)[0]
    assert isinstance(config, GraphClassifierModelConfig)
    assert config.post_mp_layer.dims in ([64, 32], [32, 16])
    assert config.mp_layers[0].use_batchnorm is True
    assert config.post_mp_layer.use_batchnorm is True


def test_train_config_validity(mock_design_space):
    config = generate_train_config(mock_design_space.train, k=1)[0]
    assert isinstance(config, TrainConfig)
    assert config.optim.optimizer == "adam"
    assert config.max_epoch in (10, 100)


def test_data_loader_config_validity(mock_design_space):
    config = generate_data_loader_config(mock_design_space.data_loader, k=1)[0]
    assert isinstance(config, DataLoaderConfig)
    assert config.graph_const in ["knn", "radius"]
