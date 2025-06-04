import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from stgym.config_schema import ExperimentConfig, MLFlowConfig, ResourceConfig
from stgym.data_loader import STDataModule
from stgym.design_space.schema import DesignSpace
from stgym.rct.exp_gen import generate_experiment_configs
from stgym.tl_model import STGymModule
from stgym.train import train
from stgym.utils import RayProgressBar, create_mlflow_experiment


def run_exp(exp_cfg: ExperimentConfig, mlflow_cfg: MLFlowConfig):
    data_module = STDataModule(exp_cfg.task, exp_cfg.data_loader)

    if exp_cfg.task.type in ("node-classification", "graph-classification"):
        dim_out = exp_cfg.task.num_classes
        if exp_cfg.task == "graph-classification" and dim_out == 2:
            dim_out -= 1
    else:
        # for clustering, dim_out is specified by the pooling operation
        dim_out = None

    model_module = STGymModule(
        dim_in=data_module.num_features,
        dim_out=dim_out,
        model_cfg=exp_cfg.model,
        train_cfg=exp_cfg.train,
        task_cfg=exp_cfg.task,
    )

    if exp_cfg.group_id is not None:
        mlflow_cfg = mlflow_cfg.model_copy()
        mlflow_cfg.tags = {
            "group_id": str(exp_cfg.group_id),
            "task_type": exp_cfg.task.type,
        }

    train(
        model_module,
        data_module,
        exp_cfg.train,
        mlflow_cfg,
        tl_train_config={
            "log_every_n_steps": 10,
            # simplify the logging
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "enable_checkpointing": False,
        },
    )

    return True


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    design_space = DesignSpace.model_validate(cfg.design_space)
    mlflow_cfg = MLFlowConfig.model_validate(cfg.mlflow)

    # create the experiment before runs start, to avoid multi-thread competition
    create_mlflow_experiment(mlflow_cfg.experiment_name)

    resource_cfg = ResourceConfig.model_validate(cfg.resource)
    ray.init(num_cpus=resource_cfg.num_cpus, num_gpus=resource_cfg.num_cpus)

    exp_cfgs = generate_experiment_configs(
        design_space,
        cfg.design_dimension,
        cfg.design_choices,
        cfg.sample_size,
        cfg.random_seed,
    )
    ray_run_exp = ray.remote(run_exp).options(
        num_cpus=resource_cfg.num_cpus_per_trial,
        num_gpus=resource_cfg.num_gpus_per_trial,
    )
    promises = [ray_run_exp.remote(exp_cfg, mlflow_cfg) for exp_cfg in exp_cfgs]

    RayProgressBar.show(promises)

    results = ray.get(promises)
    print(f"results: {results}")
    ray.shutdown()


if __name__ == "__main__":
    main()
