import os

os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from stgym.config_schema import MLFlowConfig, ResourceConfig
from stgym.design_space.schema import TaskReprDesignSpace
from stgym.rct import run_exp
from stgym.task_repr import make_exp_config, sample_designs
from stgym.utils import DatasetLoadGate, RayProgressBar, create_mlflow_experiment


@hydra.main(config_path="./conf", config_name="task_repr_config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    space = TaskReprDesignSpace.model_validate(
        OmegaConf.to_container(cfg.task_repr.design_space, resolve=True)
    )
    task_type: str = cfg.task_repr.task_type
    datasets: list[str] = list(cfg.task_repr.datasets)

    res_cfg = ResourceConfig(**cfg.resource)
    mlflow_cfg = MLFlowConfig(**cfg.mlflow)

    if res_cfg.omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(res_cfg.omp_num_threads)
        torch.set_num_threads(res_cfg.omp_num_threads)

    create_mlflow_experiment(mlflow_cfg.experiment_name)

    if not ray.is_initialized():
        ray.init(num_cpus=res_cfg.num_cpus, num_gpus=res_cfg.num_gpus)

    gated_datasets = frozenset(cfg.gated_datasets)
    if gated_datasets:
        DatasetLoadGate.options(name="dataset_load_gate").remote(max_concurrent=1)

    designs = sample_designs(space, task_type, n=cfg.n_designs, seed=cfg.random_seed)

    promises = []
    for dataset_name in datasets:
        for design_id, partial in designs:
            exp_cfg = make_exp_config(partial, dataset_name)

            run_exp_remote = ray.remote(
                num_cpus=res_cfg.num_cpus_per_trial,
                num_gpus=res_cfg.num_gpus_per_trial,
                max_calls=1,
            )(run_exp)
            promises.append(
                run_exp_remote.remote(
                    exp_cfg,
                    mlflow_cfg,
                    metadata_for_tag={
                        "design_id": str(design_id),
                        "task_repr_sweep": "true",
                    },
                )
            )

    total = len(promises)
    print(
        f"Total experiments: {total} ({cfg.n_designs} designs × {len(datasets)} tasks)"
    )

    if promises:
        RayProgressBar.show(promises)
        ray.get(promises)

    ray.shutdown()


if __name__ == "__main__":
    main()
