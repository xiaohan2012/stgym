import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from stgym.config_schema import MLFlowConfig, ResourceConfig
from stgym.design_space.schema import DesignSpace
from stgym.mem_utils import estimate_memory_usage
from stgym.rct import generate_experiment_configs, run_exp
from stgym.utils import RayProgressBar, create_mlflow_experiment


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

    # create the experiment if needed, before experiment starts
    # to avoid duplicate experiments being created
    create_mlflow_experiment(mlflow_cfg.experiment_name)

    if not ray.is_initialized():
        ray.init(num_cpus=res_cfg.num_cpus, num_gpus=res_cfg.num_gpus)

    configs = generate_experiment_configs(
        design_space,
        design_dimension,
        design_chocies,
        cfg.sample_size,
        cfg.random_seed,
    )

    # Calculate GPU requirements for each experiment
    gpu_memory_gb = res_cfg.gpu_memory_gb
    print(
        f"Estimating GPU requirements for {len(configs)} experiments (assuming {gpu_memory_gb}GB per GPU)..."
    )

    # Launch experiments with dynamic GPU allocation
    promises = []
    skipped_exp = 0

    for i, exp_cfg in enumerate(configs):
        gpu_ratio = estimate_gpu_requirements(exp_cfg, gpu_memory_gb)

        if gpu_ratio > 1:
            # Skip experiments that exceed GPU memory capacity
            print(
                f"Experiment {i+1}/{len(configs)}: SKIPPED - memory exceeds GPU capacity"
            )
            skipped_exp += 1
            continue

        print(f"Experiment {i+1}/{len(configs)}: estimated {gpu_ratio:.3f} GP(s)")

        run_exp_remote = ray.remote(run_exp).options(
            num_cpus=res_cfg.num_cpus_per_trial, num_gpus=gpu_ratio
        )
        promises.append(run_exp_remote.remote(exp_cfg, mlflow_cfg))
    # Print summary
    total_exp = len(configs)
    exp_to_execute = len(promises)
    print(f"\n🎯 Execution Summary:")
    print(f"   Total experiments: {total_exp}")
    print(f"   To execute: {exp_to_execute}")
    print(f"   Skipped (memory exceeded): {skipped_exp}")

    if promises:
        RayProgressBar.show(promises)
        ray.get(promises)

    ray.shutdown()


if __name__ == "__main__":
    main()
