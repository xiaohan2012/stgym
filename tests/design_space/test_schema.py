import pytest

from stgym.design_space.schema import (
    DataLoaderSpace,
    DesignSpace,
    ModelSpace,
    TaskSpace,
    TrainSpace,
)
from stgym.utils import load_yaml


@pytest.mark.parametrize("name", ["node-clf", "graph-clf"])
def test_basic(name):
    data = load_yaml(f"./tests/data/design-space-{name}.yaml")

    ModelSpace.model_validate(data["model"])
    TrainSpace.model_validate(data["train"])
    TaskSpace.model_validate(data["task"])
    DataLoaderSpace.model_validate(data["data_loader"])

    DesignSpace.model_validate(data)
