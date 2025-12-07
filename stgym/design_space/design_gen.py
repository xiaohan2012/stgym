import random

import numpy as np
import pydash as _
from pydantic import BaseModel

from stgym.config_schema import (
    DataLoaderConfig,
    ExperimentConfig,
    GraphClassifierModelConfig,
    MessagePassingConfig,
    NodeClassifierModelConfig,
    PostMPConfig,
    TaskConfig,
    TaskType,
    TrainConfig,
)
from stgym.data_loader.ds_info import get_info
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
    model_designs = generate_model_config(space.task.type, space.model, k, seed)
    train_designs = generate_train_config(space.train, k, seed)
    task_designs = generate_task_config(space.task, k, seed)
    dl_designs = generate_data_loader_config(space.data_loader, k, seed)
    return [
        ExperimentConfig(model=m, train=tr, task=ta, data_loader=dl)
        for m, tr, ta, dl in zip(model_designs, train_designs, task_designs, dl_designs)
    ]


def sample_across_dimensions(
    space: ModelSpace | TrainSpace | TaskSpace | DataLoaderSpace,
    seed: int | np.int_ = None,
):
    random.seed(int(seed) if seed is not None else None)
    ret = {}
    zipped_fields = space.zip_
    zip_values = [getattr(space, f) for f in zipped_fields]

    if zipped_fields:
        # ensure that the length of the values in zipped fields match
        assert (
            len(_.uniq(_.map_(zip_values, len))) == 1
        ), f"Array length mismatch: {zip_values}"

        sampled_zip_values = random.choice(_.zip_(*zip_values))
        ret = dict(zip(zipped_fields, sampled_zip_values))
    else:
        ret = {}

    remaining_fields = _.pull_all(_.keys(space.model_fields), zipped_fields + ["zip_"])

    for dimension in remaining_fields:
        val = getattr(space, dimension)
        if isinstance(val, list):
            ret[dimension] = random.choice(val)
        elif isinstance(val, BaseModel):
            ret[dimension] = sample_across_dimensions(val, seed=seed)
        else:
            ret[dimension] = val
    return ret


def generate_model_config(
    task_type: TaskType, space: ModelSpace, k: int = 1, seed: int = None
) -> list[GraphClassifierModelConfig | NodeClassifierModelConfig]:
    seeds = rand_ints(k, seed=seed)
    ret = []
    for i in range(k):
        values = sample_across_dimensions(space, seed=seeds[i])
        mp_layers = [
            MessagePassingConfig(**values) for j in range(values["num_mp_layers"])
        ]
        kwargs = {"mp_layers": mp_layers, "normalize_adj": values["normalize_adj"]}
        if values["post_mp_dims"]:
            dims = _.map_(values["post_mp_dims"].split(","), lambda s: int(s.strip()))
            kwargs["post_mp_layer"] = PostMPConfig(dims=dims, **values)

        if values["global_pooling"]:
            kwargs["global_pooling"] = values["global_pooling"]

        if task_type == "graph-classification":
            ret.append(GraphClassifierModelConfig(**kwargs))
        elif task_type == "node-classification":
            ret.append(NodeClassifierModelConfig(**kwargs))
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
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
        ds_info = get_info(values["dataset_name"])
        ret.append(TaskConfig(**values, num_classes=ds_info.get("num_classes")))
    return ret


def generate_data_loader_config(
    space: DataLoaderSpace, k: int = 1, seed: int = None
) -> list[DataLoaderConfig]:
    ret = []
    random.seed(seed)
    seeds = rand_ints(k, seed=seed)
    for i in range(k):
        # conditional sample knn_k or radius_ratio depending on the value of graph_const
        if isinstance(space.graph_const, (list, tuple)):
            graph_const = random.choice(space.graph_const)
        else:
            graph_const = space.graph_const
        space_cp = space.model_copy()
        if graph_const == "knn":
            space_cp.graph_const = "knn"
            space_cp.radius_ratio = None
        else:
            space_cp.graph_const = "radius"
            space_cp.knn_k = None
        values = sample_across_dimensions(space_cp, seed=seeds[i])
        ret.append(DataLoaderConfig(**values))
    return ret
