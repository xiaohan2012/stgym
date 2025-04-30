import random

import pydash as _
from pydantic import BaseModel
from stgym.config_schema import (
    MessagePassingConfig,
    ModelConfig,
    PostMPConfig,
    TrainConfig,
    TaskConfig,
    DataLoaderConfig,
    ExperimentConfig
)
from stgym.design_space.schema import (
    DesignSpace,
    ModelSpace,
    TaskSpace,
    TrainSpace,
    DataLoaderSpace,
)


def generate_experiment(space: DesignSpace, k: int = 1) -> list[ExperimentConfig]:
    model_designs = generate_model_config(space.model, k)
    train_designs = generate_train_config(space.train, k)
    task_designs = generate_task_config(space.task, k)
    dl_designs = generate_data_loader_config(space.data_loader, k)
    return [
        ExperimentConfig(model=m, train=tr, task=ta, data_loader=dl)
        for m, tr, ta, dl in zip(model_designs, train_designs, task_designs, dl_designs)
    ]


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
            MessagePassingConfig(**values) for j in range(values["num_mp_layers"])
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
    ret = []
    for i in range(k):
        values = sample_across_dimensions(space)
        ret.append(TrainConfig(**values))
    return ret


def generate_task_config(space: TaskSpace, k: int = 1) -> list[TaskConfig]:
    ret = []
    for i in range(k):
        values = sample_across_dimensions(space)
        # TODO: populate task-specific evaluation metrics
        ret.append(TaskConfig(**values, type="graph-classification"))
    return ret


def generate_data_loader_config(
    space: DataLoaderSpace, k: int = 1
) -> list[DataLoaderConfig]:
    ret = []
    for i in range(k):
        values = sample_across_dimensions(space)
        ret.append(DataLoaderConfig(**values))
    return ret
