import pytest
from stgym.design_space.schema import DesignSpace, ModelSpace, TaskSpace, TrainSpace
from stgym.design_space.design_gen import (
    sample_across_dimensions,
    generate_design,
    generate_model_config,
    generate_train_config,
    generate_task_config,
)
from stgym.utils import load_yaml
from stgym.config_schema import (
    ModelConfig,
    TrainConfig,
    DataLoaderConfig,
    MessagePassingConfig,
)


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
            post_mp_dims=['64,32', '32, 16'],
        )
        design = sample_across_dimensions(space)
        assert design["num_mp_layers"] == 1

        assert design["normalize_adj"] is True
        assert design["pooling"]["type"] == "dmon"
        assert design["pooling"]["n_clusters"] in (10, 20)
        assert design["layer_type"] == "ginconv"

        assert design["global_pooling"] == "mean"

        assert design["post_mp_dims"] in ['64,32', '32, 16']

@pytest.mark.parametrize("k", [1, 2, 3])
def test_generate_design(k, mock_design_space):

    designs = generate_design(mock_design_space, k=k)
    assert len(designs) == k


def test_design_validity(mock_design_space):
    design = generate_design(mock_design_space, k=1)[0]
    pass


def test_model_config_validity(mock_design_space):
    config = generate_model_config(mock_design_space.model, k=1)[0]
    assert isinstance(config, ModelConfig)
    assert config.post_mp_layer.dims in ([64, 32], [32, 16])
    assert config.mp_layers[0].use_batchnorm is True
    assert config.post_mp_layer.use_batchnorm is True
    
