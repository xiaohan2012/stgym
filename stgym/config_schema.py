from typing import Literal, Optional

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

ActivationType = Literal["prelu", "relu", "swish"]
AggregationType = Literal["mean", "sum", "max"]
PoolingType = Literal["mincut", "sum", "max", "dmod"]
StageType = Literal["skipsum", "skipconcat"]


class LayerConfig(BaseModel):
    dim_inner: PositiveInt = 256
    n_layers: PositiveInt = 2

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
    layer_type: Literal["gcnconv", "generalconv"] = "gcnconv"
    normalize_adj: bool = False
    agg: Optional[AggregationType] = "mean"


class PostMPConfig(LayerConfig):
    graph_pooling: Optional[PoolingType] = "mincut"


class MemoryConfig(BaseModel):
    inplace: bool = False


class InterLayerConfig(BaseModel):
    stage_type: Optional[StageType] = "skipconcat"


class Config(BaseModel):
    mp: MessagePassingConfig
    post_mp: PostMPConfig
    mem: MemoryConfig
    inter_layer: InterLayerConfig

    # TOOD: validate the when skipsim is used, the dim_in and dim_inner should be the same
