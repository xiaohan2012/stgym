from typing import Optional

from pydantic import BaseModel, PositiveFloat, PositiveInt

from stgym.config_schema import (
    ActivationType,
    GlobalPoolingType,
    GraphConstructionApproach,
    HierarchicalPoolingType,
    LayerType,
    LRSchedulerType,
    OptimizerType,
    TaskType,
)
from stgym.utils import YamlLoaderMixin


class ModelWithZip(BaseModel):
    zip_: Optional[list[str]] = []  # used to specify the fields to be 'zipped' over


class ModelSpace(ModelWithZip):
    class PoolingSpace(ModelWithZip):
        type: HierarchicalPoolingType | list[HierarchicalPoolingType]
        n_clusters: PositiveInt | list[PositiveInt]

    num_mp_layers: PositiveInt | list[PositiveInt]

    # not null only for graph classification
    global_pooling: Optional[GlobalPoolingType | list[GlobalPoolingType]] = None

    normalize_adj: bool | list[bool]
    layer_type: LayerType | list[LayerType]
    dim_inner: PositiveInt | list[PositiveInt]
    act: ActivationType | list[ActivationType]
    use_batchnorm: bool | list[bool]
    pooling: PoolingSpace | None

    # temporary hack, which stores a tuple of ints as a comma-separated string
    # because yaml does not distinguish tuple from list
    post_mp_dims: Optional[str | list[str]] = None


class TrainSpace(ModelWithZip):
    class OptimSpace(ModelWithZip):
        optimizer: OptimizerType | list[OptimizerType]
        base_lr: PositiveFloat | list[PositiveFloat]

    class LRSchedulerSpace(ModelWithZip):
        type: LRSchedulerType | list[LRSchedulerType]

    optim: OptimSpace
    lr_schedule: LRSchedulerSpace
    max_epoch: PositiveInt | list[PositiveInt]


class DataLoaderSpace(ModelWithZip):
    graph_const: GraphConstructionApproach | list[GraphConstructionApproach]
    knn_k: PositiveInt | list[PositiveInt]
    batch_size: PositiveInt | list[PositiveInt]


class TaskSpace(ModelWithZip):
    dataset_name: str | list[str]
    type: TaskType  # list[TaskType], only one type can be specified in one config


class DesignSpace(ModelWithZip, YamlLoaderMixin):
    model: ModelSpace
    train: TrainSpace
    task: TaskSpace
    data_loader: DataLoaderSpace
