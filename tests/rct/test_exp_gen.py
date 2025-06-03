import pydash as _
import pytest

from stgym.rct.exp_gen import RCTConfig, generate_experiment_configs, load_rct_config

from ..utils import RANDOM_SEEDS


@pytest.fixture
def cfg() -> RCTConfig:
    return load_rct_config("./tests/data/controlled-randomized-experiment-example.yaml")


def test_basic(cfg):
    assert cfg.design_space_source.is_absolute()


@pytest.mark.parametrize("k", [1, 2, 3])
def test_generation(cfg, k):
    cfg.sample_size = k
    exp_cfgs = generate_experiment_configs(cfg)

    assert len(exp_cfgs) == k * len(cfg.design_choices)
    exp_cfg_dicts = _.map_(exp_cfgs, lambda x: x.model_dump())

    assert _.sort(
        _.map_(exp_cfg_dicts, lambda x: _.get(x, "model.mp_layers.0.use_batchnorm"))
    ) == ([False] * k + [True] * k)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_config_equivalence(cfg, seed):
    """ensure that there are a 'pair' of identical experiments except on the design dimension to modify"""

    cfg.sample_size = 1
    cfg.random_seed = seed
    exp_cfgs = generate_experiment_configs(cfg)
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


def test_on_group_ids(cfg):
    cfg.sample_size = 10
    exp_cfgs = generate_experiment_configs(cfg)
    choice_multiplicity = 2  # true or false on model.use_batchnorm
    for gid in range(cfg.sample_size):
        for i in range(choice_multiplicity):
            assert exp_cfgs[gid + i * cfg.sample_size].group_id == gid


def test_invalid_design_dimension(cfg):
    cfg.design_dimension = "non-exisitent-choice"
    with pytest.raises(ValueError, match="Non-exisitent design dimension: .*"):
        generate_experiment_configs(cfg)
