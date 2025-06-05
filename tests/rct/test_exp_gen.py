import pydash as _
import pytest

from stgym.design_space.schema import DesignSpace
from stgym.rct.exp_gen import generate_experiment_configs

from ..utils import RANDOM_SEEDS


@pytest.fixture
def mock_design_space():
    return DesignSpace.from_yaml("./tests/data/design-space-graph-clf.yaml")


@pytest.mark.parametrize("k", [1, 2, 3])
def test_generation(mock_design_space, k):
    exp_cfgs = generate_experiment_configs(
        mock_design_space,
        design_dimension="model.use_batchnorm",
        design_choices=[True, False],
        sample_size=k,
        random_seed=123,
    )

    assert len(exp_cfgs) == k * 2
    exp_cfg_dicts = _.map_(exp_cfgs, lambda x: x.model_dump())

    assert _.sort(
        _.map_(exp_cfg_dicts, lambda x: _.get(x, "model.mp_layers.0.use_batchnorm"))
    ) == ([False] * k + [True] * k)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_config_equivalence(mock_design_space, seed):
    """ensure that there are a 'pair' of identical experiments except on the design dimension to modify"""
    exp_cfgs = generate_experiment_configs(
        mock_design_space,
        design_dimension="model.use_batchnorm",
        design_choices=[True, False],
        sample_size=1,
        random_seed=seed,
    )

    # choice at the design dimension should vary
    assert (
        exp_cfgs[0].model.mp_layers[0].use_batchnorm
        != exp_cfgs[1].model.mp_layers[0].use_batchnorm
    )
    assert (
        exp_cfgs[0].model.mp_layers[0].has_bias
        != exp_cfgs[1].model.mp_layers[0].has_bias
    )

    # overriding the values
    exp_cfgs[0].model.mp_layers[0].use_batchnorm = (
        exp_cfgs[1].model.mp_layers[0].use_batchnorm
    )
    exp_cfgs[0].model.mp_layers[0].has_bias = exp_cfgs[1].model.mp_layers[0].has_bias
    exp_cfgs[0].model.post_mp_layer.use_batchnorm = exp_cfgs[
        1
    ].model.post_mp_layer.use_batchnorm
    exp_cfgs[0].model.post_mp_layer.has_bias = exp_cfgs[1].model.post_mp_layer.has_bias


def test_on_group_ids(mock_design_space):
    k = 10
    exp_cfgs = generate_experiment_configs(
        mock_design_space,
        design_dimension="model.use_batchnorm",
        design_choices=[True, False],
        sample_size=k,
        random_seed=None,
    )
    choice_multiplicity = 2  # true or false on model.use_batchnorm
    for gid in range(k):
        for i in range(choice_multiplicity):
            assert exp_cfgs[gid + i * k].group_id == gid


def test_invalid_design_dimension(mock_design_space):
    with pytest.raises(ValueError, match="Non-exisitent design dimension: .*"):
        generate_experiment_configs(
            mock_design_space,
            design_dimension="non-exisitent-choice",
            design_choices=[True, False],
            sample_size=10,
            random_seed=None,
        )
