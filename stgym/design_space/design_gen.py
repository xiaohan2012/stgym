import random

import pydash as _
from pydantic import BaseModel
from stgym.config_schema import (
    MessagePassingConfig,
    ModelConfig,
    PostMPConfig,
    TrainConfig,
)
from stgym.design_space.schema import DesignSpace, ModelSpace, TaskSpace, TrainSpace


def generate_design(space: DesignSpace):
    pass


def sample_across_dimensions(space: ModelSpace | TrainSpace | TaskSpace):
    ret = {}
    for dimension in space.__fields__:
        val = getattr(space, dimension)
        if isinstance(val, list):
            ret[dimension] = random.choice(val)
        elif isinstance(val, BaseModel):
            ret[dimension] = sample_across_dimensions(val)
        else:
            ret[dimension] = val
    return ret


def generate_model_config(space: ModelSpace, k: int = 1) -> list[ModelConfig]:
    ret = []
    for i in range(k):
        values = sample_across_dimensions(space)
        mp_layers = [
            MessagePassingConfig(
                **values
                # **_.omit(values, ["num_mp_layers", "normalize_adj", "global_pooling"])
            )
            for j in range(values["num_mp_layers"])
        ]
        post_mp_dims = _.map_(
            values["post_mp_dims"].split(","), lambda s: int(s.strip())
        )
        print("post_mp_dims: {}".format(post_mp_dims))
        ret.append(
            ModelConfig(
                global_pooling=values["global_pooling"],
                normalize_adj=values["normalize_adj"],
                mp_layers=mp_layers,
                post_mp_layer=PostMPConfig(dims=post_mp_dims, **values),
            )
        )
    return ret


def generate_train_config(space: TrainSpace, k: int = 1) -> list[TrainConfig]:
    pass


def generate_task_config(space: TaskSpace, k: int = 1):
    pass
