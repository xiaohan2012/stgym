from typing import Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

ActivationType = Literal["prelu", "relu", "swish"]
GlobalPoolingType = Literal["mean", "sum", "max"]
HierarchicalPoolingType = Literal["mincut", "dmon"]  # hierarchical pooling
StageType = Literal["skipsum", "skipconcat"]
LayerType = Literal["gcnconv", "ginconv", "sageconv", "linear"]
OptimizerType = Literal["sgd", "adam"]


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
    pass


class MemoryConfig(BaseModel):
    inplace: bool = False


class InterLayerConfig(BaseModel):
    stage_type: Optional[StageType] = "skipconcat"


class ModelConfig(BaseModel):
    layers: list[MessagePassingConfig | PostMPConfig | LayerConfig]
    # mem: Optional[MemoryConfig] = MemoryConfig()

    @property
    def n_layers(self):
        return len(self.layers)


class OptimizerConfig(BaseModel):
    optimizer: OptimizerType = "adam"
    base_lr: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.9
