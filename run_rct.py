import os

# Disable CUDA memory caching to prevent NVML errors on virtual GPU environments (#48)
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import hydra
import pydash as pyd
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from stgym.config_schema import MLFlowConfig, ResourceConfig
from stgym.design_space.schema import DesignSpace
from stgym.mem_utils import estimate_memory_usage
from stgym.rct import generate_experiment_configs, run_exp
from stgym.utils import DatasetLoadGate, RayProgressBar, create_mlflow_experiment


def estimate_gpu_requirements(exp_cfg, gpu_memory_gb: float) -> float:
    """Estimate how many GPUs needed for an experiment based on memory usage.

    Args:
        exp_cfg: Experiment configuration
        gpu_memory_gb: Available memory per GPU in GB

    Returns:
        Float representing GPU fraction needed
    """
    total_memory_gb, _ = estimate_memory_usage(exp_cfg)

    return total_memory_gb / gpu_memory_gb


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    design_space = DesignSpace.model_validate(cfg.design_space)
    design_dimension = cfg.design_dimension
    design_chocies = cfg.design_choices
    res_cfg = ResourceConfig(**cfg.resource)
    mlflow_cfg = MLFlowConfig(**cfg.mlflow)

    if res_cfg.omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(res_cfg.omp_num_threads)
        torch.set_num_threads(res_cfg.omp_num_threads)

    # create the experiment if needed, before experiment starts
    # to avoid duplicate experiments being created
    create_mlflow_experiment(mlflow_cfg.experiment_name)

    if not ray.is_initialized():
        ray.init(num_cpus=res_cfg.num_cpus, num_gpus=res_cfg.num_gpus)

    gated_datasets = frozenset(cfg.gated_datasets)
    if gated_datasets:
        DatasetLoadGate.options(name="dataset_load_gate").remote(max_concurrent=1)

    configs = generate_experiment_configs(
        design_space,
        design_dimension,
        design_chocies,
        cfg.sample_size,
        cfg.random_seed,
    )

    promises = []

    for i, exp_cfg in enumerate(configs):
        # Use configured GPU allocation instead of memory-based estimation
        gpu_allocation = res_cfg.num_gpus_per_trial

        # max_calls=1: recycle worker after each task to prevent heap fragmentation
        # Without this, RSS grows ~70MB/task as pymalloc arenas fragment
        # See PR #143 for diagnostic data and root cause analysis
        run_exp_remote = ray.remote(
            num_cpus=res_cfg.num_cpus_per_trial,
            num_gpus=gpu_allocation,
            max_calls=1,
        )(run_exp)
        promises.append(
            run_exp_remote.remote(
                exp_cfg,
                mlflow_cfg,
                gated_datasets=gated_datasets,
                # add additional mlflow tags
                metadata_for_tag={
                    "design_dimension": design_dimension,
                    "design_chocies": "|".join(pyd.map_(design_chocies, str)),
                },
            )
        )
    # Print summary
    total_exp = len(promises)
    print(f"Total experiments to run: {total_exp}")

    if promises:
        RayProgressBar.show(promises)

    ray.shutdown()


if __name__ == "__main__":
    main()
