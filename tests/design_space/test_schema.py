from stgym.utils import load_yaml
from stgym.design_space.schema import ModelSpace, TrainSpace, TaskSpace, DesignSpace


def test_basic():
    data = load_yaml("./tests/data/design-space-example.yaml")

    ModelSpace.model_validate(data["model"])
    TrainSpace.model_validate(data["train"])
    TaskSpace.model_validate(data["task"])

    DesignSpace.model_validate(data)
