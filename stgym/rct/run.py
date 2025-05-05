import ray

from stgym.config_schema import ExperimentConfig, MLFlowConfig
from stgym.data_loader import STDataModule
from stgym.rct.exp_gen import generate_experiment_configs, load_rct_config
from stgym.tl_model import STGymModule
from stgym.train import train


# @ray.remote(num_cpus=1, num_gpus=0.1)
@ray.remote(num_cpus=1)
def run_exp(exp_cfg: ExperimentConfig, mlflow_cfg: MLFlowConfig):
    data_module = STDataModule(exp_cfg.task, exp_cfg.data_loader)
    model_module = STGymModule(
        dim_in=data_module.num_features,
        dim_out=1,  # 1 for binary classification
        model_cfg=exp_cfg.model,
        train_cfg=exp_cfg.train,
    )
    train(
        model_module,
        data_module,
        exp_cfg.train,
        mlflow_cfg,
        tl_train_config={"log_every_n_steps": 10},
    )

    return True


def main():
    rct_config_path = "./configs/rct/bn.yaml"
    mlflow_cfg_path = "./configs/mlflow.yaml"

    rct_config = load_rct_config(rct_config_path)

    mlflow_cfg = MLFlowConfig.from_yaml(mlflow_cfg_path)
    mlflow_cfg.experiment_name = rct_config.experiment_name

    exp_cfgs = generate_experiment_configs(rct_config)
    promises = [run_exp.remote(exp_cfg, mlflow_cfg) for exp_cfg in exp_cfgs]
    results = ray.get(promises)
    print(f"results: {results}")


if __name__ == "__main__":
    main()
