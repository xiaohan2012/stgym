import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from stgym.config_schema import MLFlowConfig, ResourceConfig
from stgym.design_space.schema import DesignSpace
from stgym.rct import generate_experiment_configs, run_exp
from stgym.utils import RayProgressBar, create_mlflow_experiment


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

    run_exp_remote = ray.remote(run_exp).options(
        num_cpus=res_cfg.num_cpus_per_trial, num_gpus=res_cfg.num_gpus_per_trial
    )

    promises = [run_exp_remote.remote(exp_cfg, mlflow_cfg) for exp_cfg in configs]
    RayProgressBar.show(promises)
    ray.get(promises)

    ray.shutdown()


if __name__ == "__main__":
    main()
