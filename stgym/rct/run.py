import ray

from stgym.config_schema import ExperimentConfig
from stgym.data_loader import STDataModule
from stgym.rct.exp_gen import generate_experiment_configs, load_config
from stgym.tl_model import STGymModule
from stgym.train import train


@ray.remote(num_cpus=1, num_gpus=0.1)
def run_exp(cfg: ExperimentConfig):
    data_module = STDataModule(cfg.task, cfg.data_loader)
    model_module = STGymModule(
        dim_in=data_module.num_features,
        dim_out=1,  # 1 for binary classification
        model_cfg=cfg.model,
        train_cfg=cfg.train,
    )
    train(
        model_module, data_module, cfg.train, trainer_config={"log_every_n_steps": 10}
    )
    return True


def main():
    rct_config_path = "./configs/rct/bn.yaml"
    rct_config = load_config(rct_config_path)

    exp_cfgs = generate_experiment_configs(rct_config)
    promises = [run_exp.remote(cfg) for cfg in exp_cfgs]
    results = ray.get(promises)
    print(f"results: {results}")


if __name__ == "__main__":
    main()
