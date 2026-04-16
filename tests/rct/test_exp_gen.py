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


@pytest.mark.parametrize(
    "design_dimension, design_choices, cfg_accessor",
    [
        (
            "model.layer_type",
            ["gcnconv", "ginconv", "sageconv"],
            lambda cfg: cfg.model.mp_layers[0].layer_type,
        ),
        (
            "model.global_pooling",
            ["mean", "max"],
            lambda cfg: cfg.model.global_pooling,
        ),
        (
            "model.normalize_adj",
            [True, False],
            lambda cfg: cfg.model.mp_layers[0].normalize_adj,
        ),
    ],
)
def test_new_design_dimensions(
    mock_design_space, design_dimension, design_choices, cfg_accessor
):
    """Verify generate_experiment_configs correctly handles a design dimension."""
    k = 10
    exp_cfgs = generate_experiment_configs(
        mock_design_space,
        design_dimension=design_dimension,
        design_choices=design_choices,
        sample_size=k,
        random_seed=42,
    )

    assert len(exp_cfgs) == k * len(design_choices)
    actual_choices = _.sort(_.uniq(_.map_(exp_cfgs, cfg_accessor)))
    assert actual_choices == _.sort(design_choices)


def test_invalid_design_dimension(mock_design_space):
    with pytest.raises(ValueError, match="Non-exisitent design dimension: .*"):
        generate_experiment_configs(
            mock_design_space,
            design_dimension="non-exisitent-choice",
            design_choices=[True, False],
            sample_size=10,
            random_seed=None,
        )


@pytest.mark.parametrize(
    "locked_graph_const, design_dimension, design_choices",
    [
        ("knn", "data_loader.knn_k", [10, 20, 30]),
        ("radius", "data_loader.radius_ratio", [0.05, 0.075, 0.1]),
    ],
)
def test_no_wasted_runs_when_graph_const_is_locked(
    mock_design_space, locked_graph_const, design_dimension, design_choices
):
    """Locking graph_const ensures every run exercises the intended design dimension.

    Reproduces issue #87: when graph_const is free to vary, ~50% of knn/radius
    experiment runs sample the wrong graph construction method, making the design
    dimension (knn_k or radius_ratio) irrelevant for those runs.
    """
    ds = _.set_(mock_design_space, "data_loader.graph_const", locked_graph_const)
    exp_cfgs = generate_experiment_configs(
        ds,
        design_dimension=design_dimension,
        design_choices=design_choices,
        sample_size=10,
        random_seed=42,
    )

    for cfg in exp_cfgs:
        assert cfg.data_loader.graph_const == locked_graph_const
        if locked_graph_const == "knn":
            assert cfg.data_loader.radius_ratio is None
            assert cfg.data_loader.knn_k in design_choices
        else:
            assert cfg.data_loader.knn_k is None
            assert cfg.data_loader.radius_ratio in design_choices
