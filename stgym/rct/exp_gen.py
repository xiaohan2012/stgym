import pydash as _

from stgym.config_schema import ExperimentConfig
from stgym.design_space.design_gen import generate_experiment
from stgym.design_space.schema import DesignSpace


def generate_experiment_configs(
    design_space_template: DesignSpace,
    design_dimension: str,
    design_choices: list,
    sample_size: int,
    random_seed: int = None,
) -> list[ExperimentConfig]:
    if _.has(design_space_template.model_dump(), design_dimension):
        exp_cfgs_by_design_choice = [
            generate_experiment(
                _.set_(design_space_template, design_dimension, choice),
                k=sample_size,
                seed=random_seed,
            )
            for choice in design_choices
        ]
        for group_id, grouped_exps in enumerate(zip(*exp_cfgs_by_design_choice)):
            for exp in grouped_exps:
                exp.group_id = group_id

        return _.flatten(exp_cfgs_by_design_choice)
    else:
        raise ValueError(f"Non-exisitent design dimension: '{design_dimension}'")
