import pydash as _
import pytest

from stgym.rct.exp_gen import RCTConfig, generate_experiments
from stgym.utils import load_yaml


@pytest.fixture
def cfg():
    config_file = "./tests/data/controlled-randomized-experiment-example.yaml"
    data = load_yaml(config_file)

    return RCTConfig.model_validate(data | {"config_file": config_file})


def test_basic(cfg):
    assert cfg.design_space_source.is_absolute()


@pytest.mark.parametrize("k", [1, 2, 3])
def test_generation(cfg, k):
    exp_cfgs = generate_experiments(cfg, k=k)

    assert len(exp_cfgs) == k * len(cfg.design_choices)
    exp_cfg_dicts = _.map_(exp_cfgs, lambda x: x.model_dump())

    assert _.sort(
        _.map_(exp_cfg_dicts, lambda x: _.get(x, "model.mp_layers.0.use_batchnorm"))
    ) == ([False] * k + [True] * k)


def test_invalid_design_dimension(cfg):
    cfg.design_dimension = "non-exisitent-choice"
    with pytest.raises(ValueError, match="Non-exisitent design dimension: .*"):
        exp_cfgs = generate_experiments(cfg, k=1)
