import pydash as _
import pytest

from stgym.rct.exp_gen import generate_experiments, load_config


@pytest.fixture
def cfg():
    return load_config("./tests/data/controlled-randomized-experiment-example.yaml")


def test_basic(cfg):
    assert cfg.design_space_source.is_absolute()


@pytest.mark.parametrize("k", [1, 2, 3])
def test_generation(cfg, k):
    cfg.sample_size = k
    exp_cfgs = generate_experiments(cfg)

    assert len(exp_cfgs) == k * len(cfg.design_choices)
    exp_cfg_dicts = _.map_(exp_cfgs, lambda x: x.model_dump())

    assert _.sort(
        _.map_(exp_cfg_dicts, lambda x: _.get(x, "model.mp_layers.0.use_batchnorm"))
    ) == ([False] * k + [True] * k)


def test_invalid_design_dimension(cfg):
    cfg.design_dimension = "non-exisitent-choice"
    with pytest.raises(ValueError, match="Non-exisitent design dimension: .*"):
        generate_experiments(cfg)
