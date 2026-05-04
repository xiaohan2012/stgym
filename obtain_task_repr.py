import os
from datetime import datetime

os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from stgym.config_schema import (
    ExperimentConfig,
    MLFlowConfig,
    ResourceConfig,
    TaskConfig,
)
from stgym.data_loader.ds_info import get_info
from stgym.design_space.schema import TaskReprDesignSpace
from stgym.rct import run_exp
from stgym.task_repr import sample_task_free_designs
from stgym.utils import DatasetLoadGate, RayProgressBar, create_mlflow_experiment


@hydra.main(
    config_path="./conf", config_name="obtain_task_repr_node_clf", version_base=None
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    task_type = cfg.task_type
    space = TaskReprDesignSpace.model_validate(OmegaConf.to_container(cfg.design_space))
    res_cfg = ResourceConfig(**cfg.resource)
    mlflow_cfg = MLFlowConfig(**cfg.mlflow)

    if res_cfg.omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(res_cfg.omp_num_threads)
        torch.set_num_threads(res_cfg.omp_num_threads)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_type_slug = task_type.replace("-", "_")
    exp_name = f"task-repr-{task_type_slug}-{timestamp}"
    mlflow_cfg.experiment_name = exp_name
    create_mlflow_experiment(exp_name)

    if not ray.is_initialized():
        ray.init(num_cpus=res_cfg.num_cpus, num_gpus=res_cfg.num_gpus)

    gated_datasets = frozenset(cfg.gated_datasets)
    if gated_datasets:
        DatasetLoadGate.options(name="dataset_load_gate").remote(max_concurrent=1)

    designs = sample_task_free_designs(task_type, space, cfg.n_designs, cfg.seed)
    dataset_infos = {name: get_info(name) for name in cfg.tasks}
    run_exp_remote = ray.remote(
        num_cpus=res_cfg.num_cpus_per_trial,
        num_gpus=res_cfg.num_gpus_per_trial,
        max_calls=1,
    )(run_exp)

    promises = []
    for design in designs:
        for dataset_name in cfg.tasks:
            task_cfg = TaskConfig(
                dataset_name=dataset_name,
                type=task_type,
                num_classes=dataset_infos[dataset_name]["num_classes"],
            )
            exp_cfg = ExperimentConfig(
                task=task_cfg,
                model=design.model,
                train=design.train,
                data_loader=design.data_loader.model_copy(deep=True),
            )
            promises.append(
                run_exp_remote.remote(
                    exp_cfg,
                    mlflow_cfg,
                    gated_datasets=gated_datasets,
                    metadata_for_tag={
                        "design_id": str(design.design_id),
                        "dataset_name": dataset_name,
                    },
                )
            )

    print(f"Total experiments to run: {len(promises)}")
    if promises:
        RayProgressBar.show(promises)
        ray.get(promises)

    ray.shutdown()


if __name__ == "__main__":
    main()
