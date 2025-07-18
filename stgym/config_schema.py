from typing import Literal, Optional

import pydash as _
import torch
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from stgym.utils import YamlLoaderMixin

ActivationType = Literal["prelu", "relu", "swish"]
GlobalPoolingType = Literal["mean", "sum", "max"]
HierarchicalPoolingType = Literal["mincut", "dmon"]  # hierarchical pooling
StageType = Literal["skipsum", "skipconcat"]
LayerType = Literal["gcnconv", "ginconv", "sageconv", "linear"]
OptimizerType = Literal["sgd", "adam"]
LRSchedulerType = Literal[None, "cos"]
GlobalPoolingType = Literal["max", "mean", "add"]
PostMPLayerType = Literal["mlp", "linear"]
GraphConstructionApproach = Literal["knn", "radius"]
TaskType = Literal["node-classification", "graph-classification", "node-clustering"]
EvalMetric = Literal["pr-auc", "roc-auc", "accuracy", "nmi"]


class PoolingConfig(BaseModel):
    type: HierarchicalPoolingType
    n_clusters: PositiveInt = Field(gt=1)


class LayerConfig(BaseModel):
    layer_type: LayerType

    dim_inner: PositiveInt = 256

    act: Optional[ActivationType] = "prelu"

    use_batchnorm: bool = True
    bn_eps: PositiveFloat = 1e-5
    bn_momentum: PositiveFloat = 0.1

    dropout: NonNegativeFloat = 0.0

    has_bias: bool = True
    l2norm: bool = False

    @model_validator(mode="after")
    def set_has_bias(self) -> "LayerConfig":
        self.has_bias = not self.use_batchnorm
        return self

    @property
    def has_act(self) -> bool:
        return self.act is not None


class MessagePassingConfig(LayerConfig):
    normalize_adj: bool = False

    pooling: Optional[PoolingConfig] = None

    readout: Optional[GlobalPoolingType] = "mean"

    @property
    def has_pooling(self):
        return self.pooling is not None


class PostMPConfig(LayerConfig):
    """configuration for an MLP block, placed after the message-passing layers"""

    # new field
    dims: list[int] = Field(..., min_length=1)

    # exclude irrelevant fields
    layer_type: SkipJsonSchema[LayerType] = Field(default="linear", exclude=True)
    dim_inner: SkipJsonSchema[int] = Field(default=0, exclude=True)

    def to_layer_configs(self):
        other_params = _.omit(self.model_dump(), "hidden_dims")
        return [
            LayerConfig(layer_type="linear", dim_inner=dim, **other_params)
            for dim in self.dims
        ]


class MemoryConfig(BaseModel):
    inplace: bool = False


class InterLayerConfig(BaseModel):
    stage_type: Optional[StageType] = "skipconcat"


class LRScheduleConfig(BaseModel):
    type: LRSchedulerType = None
    max_epoch: PositiveInt = 100


class OptimizerConfig(BaseModel):
    optimizer: OptimizerType = "adam"
    base_lr: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.9


class GraphClassifierModelConfig(BaseModel):
    # architecture
    mp_layers: list[MessagePassingConfig]
    global_pooling: GlobalPoolingType = "mean"
    post_mp_layer: PostMPConfig

    # misc
    mem: Optional[MemoryConfig] = MemoryConfig(inplace=False)


class ClusteringModelConfig(BaseModel):
    # architecture
    mp_layers: list[MessagePassingConfig]

    # misc
    mem: Optional[MemoryConfig] = MemoryConfig(inplace=False)


class NodeClassifierModelConfig(BaseModel):
    # architecture
    mp_layers: list[MessagePassingConfig]
    post_mp_layer: PostMPConfig

    # misc
    mem: Optional[MemoryConfig] = MemoryConfig(inplace=False)


class DataLoaderConfig(BaseModel):
    class DataSplitConfig(BaseModel):
        train_ratio: PositiveFloat
        val_ratio: PositiveFloat
        test_ratio: PositiveFloat

        @model_validator(mode="after")
        def ratios_should_sum_to_one(self) -> "LayerConfig":
            if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) >= 1e-5:
                raise ValueError("train/val/test ratios do not sum up to 1")
            return self

    graph_const: GraphConstructionApproach = "knn"
    knn_k: Optional[PositiveInt] = 10
    radius_ratio: Optional[PositiveFloat] = 0.1

    batch_size: Optional[PositiveInt] = 64
    num_workers: Optional[int] = 0  # does not suppot num_workers > 0 yet

    split: Optional[DataSplitConfig] = DataSplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    device: Optional[str] = "cpu" if not torch.cuda.is_available() else "cuda:0"


class TrainConfig(BaseModel):
    class EarlyStoppingConfig(BaseModel):
        metric: str
        mode: str = "min"

    optim: Optional[OptimizerConfig] = OptimizerConfig()
    lr_schedule: Optional[LRScheduleConfig] = LRScheduleConfig()
    max_epoch: PositiveInt

    early_stopping: Optional[EarlyStoppingConfig] = None

    @model_validator(mode="after")
    def override_scheduler_properties(self) -> Self:
        self.lr_schedule.max_epoch = self.max_epoch
        return self


class TaskConfig(BaseModel):
    dataset_name: str
    type: TaskType
    num_classes: Optional[int] = None

    @model_validator(mode="after")
    def validate_num_classes(self) -> Self:
        if self.type != "node-clustering" and self.num_classes is None:
            raise ValueError(
                "num_classes is required for node-classification and graph-classification"
            )

        return self


class DataConfig(BaseModel):
    class DataSplitConfig(BaseModel):
        train_ratio: PositiveFloat
        val_ratio: PositiveFloat
        test_ratio: PositiveFloat


class ExperimentConfig(BaseModel):
    task: TaskConfig
    data_loader: DataLoaderConfig
    model: (
        GraphClassifierModelConfig | NodeClassifierModelConfig | ClusteringModelConfig
    )
    train: TrainConfig
    group_id: Optional[int] = None


class MLFlowConfig(BaseModel, YamlLoaderMixin):
    track: Optional[bool] = True
    tracking_uri: Optional[HttpUrl] = "http://127.0.0.1:8080"
    experiment_name: Optional[str] = Field(default="test", min_length=1)
    run_name: Optional[str] = None
    tags: Optional[dict[str, str]] = None


class ResourceConfig(BaseModel, YamlLoaderMixin):
    num_cpus: Optional[PositiveInt] = None
    num_gpus: Optional[NonNegativeInt] = None
    num_cpus_per_trial: Optional[PositiveFloat] = 1.0
    num_gpus_per_trial: Optional[NonNegativeFloat] = 0.25
