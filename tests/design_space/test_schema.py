from stgym.design_space.schema import (
    DataLoaderSpace,
    DesignSpace,
    ModelSpace,
    TaskSpace,
    TrainSpace,
)
from stgym.utils import load_yaml


def test_basic():
    data = load_yaml("./tests/data/design-space-example.yaml")

    ModelSpace.model_validate(data["model"])
    TrainSpace.model_validate(data["train"])
    TaskSpace.model_validate(data["task"])
    DataLoaderSpace.model_validate(data["data_loader"])

    DesignSpace.model_validate(data)
