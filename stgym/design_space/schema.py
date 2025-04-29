from pydantic import BaseModel, PositiveInt, PositiveFloat
from stgym.config_schema import (
    GlobalPoolingType,
    LayerType,
    ActivationType,
    HierarchicalPoolingType,
    OptimizerType,
    LRSchedulerType,
    GraphConstructionApproach,
)


class ModelSpace(BaseModel):
    class PoolingSpace(BaseModel):
        type: HierarchicalPoolingType | list[HierarchicalPoolingType]
        n_clusters: PositiveInt | list[PositiveInt]

    num_layers: PositiveInt | list[PositiveInt]
    global_pooling: GlobalPoolingType | list[GlobalPoolingType]
    normalize_adj: bool | list[bool]
    layer_type: LayerType | list[LayerType]
    dim_inner: PositiveInt | list[PositiveInt]
    act: ActivationType | list[ActivationType]
    use_batchnorm: bool | list[bool]
    pooling: PoolingSpace


class TrainSpace(BaseModel):
    class OptimSpace(BaseModel):
        optimizer: OptimizerType | list[OptimizerType]
        base_lr: PositiveFloat | list[PositiveFloat]

    class LRSchedulerSpace(BaseModel):
        type: LRSchedulerType | list[LRSchedulerType]

    optim: OptimSpace
    lr_schedule: LRSchedulerSpace
    max_epoch: PositiveInt | list[PositiveInt]


class TaskSpace(BaseModel):
    dataset_name: str | list[str]
    graph_const: GraphConstructionApproach | list[GraphConstructionApproach]
    knn_k: PositiveInt | list[PositiveInt]
    batch_size: PositiveInt | list[PositiveInt]


class DesignSpace(BaseModel):
    model: ModelSpace
    train: TrainSpace
    task: TaskSpace
