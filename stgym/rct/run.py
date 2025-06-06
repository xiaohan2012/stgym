from logzero import logger
from omegaconf import OmegaConf

from stgym.config_schema import ExperimentConfig, MLFlowConfig
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule
from stgym.train import train


def run_exp(exp_cfg: ExperimentConfig, mlflow_cfg: MLFlowConfig):
    logger.debug(OmegaConf.to_yaml(exp_cfg.model_dump()))
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
