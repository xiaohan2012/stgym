import pytest

from stgym.config_schema import (
    DataLoaderConfig,
    ExperimentConfig,
    ModelConfig,
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
from stgym.design_space.schema import DesignSpace, ModelSpace
from stgym.utils import load_yaml


@pytest.fixture
def mock_design_space():
    data = load_yaml("./tests/data/design-space-example.yaml")

    return DesignSpace.model_validate(data)


class TestSampleAcrossDimensions:
    def test_basic(self):
        space = ModelSpace(
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
        design = sample_across_dimensions(space)
        assert design["num_mp_layers"] == 1

        assert design["normalize_adj"] is True
        assert design["pooling"]["type"] == "dmon"
        assert design["pooling"]["n_clusters"] in (10, 20)
        assert design["layer_type"] == "ginconv"

        assert design["global_pooling"] == "mean"

        assert design["post_mp_dims"] in ["64,32", "32, 16"]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_generate_design(k, mock_design_space):

    experiments = generate_experiment(mock_design_space, k=k)
    assert len(experiments) == k
    for exp in experiments:
        assert isinstance(exp, ExperimentConfig)


def test_model_config_validity(mock_design_space):
    config = generate_model_config(mock_design_space.model, k=1)[0]
    assert isinstance(config, ModelConfig)
    assert config.post_mp_layer.dims in ([64, 32], [32, 16])
    assert config.mp_layers[0].use_batchnorm is True
    assert config.post_mp_layer.use_batchnorm is True


def test_train_config_validity(mock_design_space):
    config = generate_train_config(mock_design_space.train, k=1)[0]
    assert isinstance(config, TrainConfig)
    assert config.optim.optimizer == "adam"
    assert config.max_epoch in (10, 100)


def test_task_config_validity(mock_design_space):
    config = generate_task_config(mock_design_space.task, k=1)[0]
    assert isinstance(config, TaskConfig)
    assert config.dataset_name in ["brca", "animal"]


def test_data_loader_config_validity(mock_design_space):
    config = generate_data_loader_config(mock_design_space.data_loader, k=1)[0]
    assert isinstance(config, DataLoaderConfig)
    assert config.graph_const in ["knn", "radius"]


@pytest.mark.parametrize("seed", [42, 123])
def test_consistency_under_fixed_random_seed(mock_design_space, seed):
    exp1 = generate_experiment(mock_design_space, k=1, seed=seed)
    exp2 = generate_experiment(mock_design_space, k=1, seed=seed)
    assert exp1 == exp2


def test_experiments_and_truly_randomized(mock_design_space):
    exp1, exp2 = generate_experiment(mock_design_space, k=2, seed=42)
    assert exp1 != exp2
