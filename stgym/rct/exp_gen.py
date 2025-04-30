from pathlib import Path
from typing import Optional

import pydash as _
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from typing_extensions import Self

from stgym.config_schema import ExperimentConfig
from stgym.design_space.design_gen import generate_experiment
from stgym.design_space.schema import DesignSpace
from stgym.utils import load_yaml


class RCTConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    design_space_source: Path
    sample_size: PositiveInt
    design_dimension: str
    design_choices: list[any]
    config_file: Optional[Path] = None

    @model_validator(mode="after")
    def expand_soruce_to_abs_path(self) -> Self:
        if not self.design_space_source.is_absolute():
            # infer the absoluate path if it is relative
            self.design_space_source = (
                self.config_file.parent / self.design_space_source
            ).absolute()
        return self


def generate_experiment_configs(cfg: RCTConfig) -> list[ExperimentConfig]:
    design_space_template = DesignSpace.model_validate(
        load_yaml(cfg.design_space_source)
    )

    if _.has(design_space_template.model_dump(), cfg.design_dimension):
        return _.flatten(
            [
                generate_experiment(
                    _.set_(design_space_template, cfg.design_dimension, choice),
                    k=cfg.sample_size,
                )
                for choice in cfg.design_choices
            ]
        )
    else:
        raise ValueError(f"Non-exisitent design dimension: '{cfg.design_dimension}'")


def load_config(config_file):
    data = load_yaml(config_file)

    return RCTConfig.model_validate(data | {"config_file": config_file})
