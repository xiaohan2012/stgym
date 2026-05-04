from dataclasses import dataclass

from stgym.config_schema import (
    DataLoaderConfig,
    GraphClassifierModelConfig,
    NodeClassifierModelConfig,
    TaskType,
    TrainConfig,
)
from stgym.design_space.design_gen import (
    generate_data_loader_config,
    generate_model_config,
    generate_train_config,
)
from stgym.design_space.schema import TaskReprDesignSpace

ModelConfig = GraphClassifierModelConfig | NodeClassifierModelConfig


@dataclass
class TaskFreeDesign:
    design_id: int
    model: ModelConfig
    train: TrainConfig
    data_loader: DataLoaderConfig


def sample_task_free_designs(
    task_type: TaskType,
    space: TaskReprDesignSpace,
    n_designs: int,
    seed: int,
) -> list[TaskFreeDesign]:
    model_configs = generate_model_config(task_type, space.model, n_designs, seed)
    train_configs = generate_train_config(space.train, n_designs, seed)
    dl_configs = generate_data_loader_config(space.data_loader, n_designs, seed)
    return [
        TaskFreeDesign(design_id=i, model=m, train=tr, data_loader=dl)
        for i, (m, tr, dl) in enumerate(zip(model_configs, train_configs, dl_configs))
    ]
