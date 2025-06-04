import pydash as _
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from stgym.config_schema import ExperimentConfig
from stgym.design_space.design_gen import generate_experiment
from stgym.design_space.schema import DesignSpace
from stgym.utils import load_yaml


class RCTConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # experiment_name: str
    # design_space_source: Path
    sample_size: PositiveInt
    design_dimension: str
    design_choices: list[any]
    # config_file: Optional[Path] = None
    random_seed: NonNegativeInt = 42

    # mlflow: Optional[MLFlowConfig] = None

    # @model_validator(mode="after")
    # def expand_soruce_to_abs_path(self) -> Self:
    #     if not self.design_space_source.is_absolute():
    #         # infer the absoluate path if it is relative
    #         self.design_space_source = (
    #             self.config_file.parent / self.design_space_source
    #         ).absolute()
    #     return self


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


def load_rct_config(config_file):
    data = load_yaml(config_file)

    return RCTConfig.model_validate(data | {"config_file": config_file})
