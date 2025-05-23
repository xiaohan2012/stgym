import random

import numpy as np
import pydash as _
from pydantic import BaseModel

from stgym.config_schema import (
    DataLoaderConfig,
    ExperimentConfig,
    GraphClassifierModelConfig,
    MessagePassingConfig,
    PostMPConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.design_space.schema import (
    DataLoaderSpace,
    DesignSpace,
    ModelSpace,
    TaskSpace,
    TrainSpace,
)
from stgym.utils import rand_ints


def generate_experiment(
    space: DesignSpace, k: int = 1, seed: int = None
) -> list[ExperimentConfig]:
    model_designs = generate_model_config(space.model, k, seed)
    train_designs = generate_train_config(space.train, k, seed)
    task_designs = generate_task_config(space.task, k, seed)
    dl_designs = generate_data_loader_config(space.data_loader, k, seed)
    return [
        ExperimentConfig(model=m, train=tr, task=ta, data_loader=dl)
        for m, tr, ta, dl in zip(model_designs, train_designs, task_designs, dl_designs)
    ]


def sample_across_dimensions(
    space: ModelSpace | TrainSpace | TaskSpace, seed: int | np.int_ = None
):
    random.seed(int(seed) if seed is not None else None)
    ret = {}
    for dimension in space.__fields__:
        val = getattr(space, dimension)
        if isinstance(val, list):
            ret[dimension] = random.choice(val)
        elif isinstance(val, BaseModel):
            ret[dimension] = sample_across_dimensions(val, seed=seed)
        else:
            ret[dimension] = val
    return ret


def generate_model_config(
    space: ModelSpace, k: int = 1, seed: int = None
) -> list[GraphClassifierModelConfig]:
    seeds = rand_ints(k, seed=seed)
    ret = []
    for i in range(k):
        values = sample_across_dimensions(space, seed=seeds[i])
        mp_layers = [
            MessagePassingConfig(**values) for j in range(values["num_mp_layers"])
        ]
        post_mp_dims = _.map_(
            values["post_mp_dims"].split(","), lambda s: int(s.strip())
        )
        ret.append(
            GraphClassifierModelConfig(
                global_pooling=values["global_pooling"],
                normalize_adj=values["normalize_adj"],
                mp_layers=mp_layers,
                post_mp_layer=PostMPConfig(dims=post_mp_dims, **values),
            )
        )
    return ret


def generate_train_config(
    space: TrainSpace, k: int = 1, seed: int = None
) -> list[TrainConfig]:
    ret = []
    seeds = rand_ints(k, seed=seed)
    for i in range(k):
        values = sample_across_dimensions(space, seed=seeds[i])
        ret.append(TrainConfig(**values))
    return ret


def generate_task_config(
    space: TaskSpace, k: int = 1, seed: int = None
) -> list[TaskConfig]:
    ret = []
    seeds = rand_ints(k, seed=seed)
    for i in range(k):
        values = sample_across_dimensions(space, seed=seeds[i])
        # TODO: populate task-specific evaluation metrics
        ret.append(TaskConfig(**values, type="graph-classification"))
    return ret


def generate_data_loader_config(
    space: DataLoaderSpace, k: int = 1, seed: int = None
) -> list[DataLoaderConfig]:
    ret = []
    seeds = rand_ints(k, seed=seed)
    for i in range(k):
        values = sample_across_dimensions(space, seed=seeds[i])
        ret.append(DataLoaderConfig(**values))
    return ret
