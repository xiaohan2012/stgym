from logzero import logger as logz_logger
from omegaconf import OmegaConf

from stgym.config_schema import ExperimentConfig, MLFlowConfig
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule
from stgym.train import train
from stgym.utils import log_experiment_config_as_artifact


def run_exp(exp_cfg: ExperimentConfig, mlflow_cfg: MLFlowConfig):
    logz_logger.debug(OmegaConf.to_yaml(exp_cfg.model_dump()))
    data_module = STDataModule(exp_cfg.task, exp_cfg.data_loader)

    if exp_cfg.task.type in ("node-classification", "graph-classification"):
        dim_out = exp_cfg.task.num_classes
        if dim_out == 2:
            dim_out -= 1  # for binary classification, output dim being 1 is enough
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
            "dataset_name": exp_cfg.task.dataset_name,
        }

    logger = mlflow_cfg.create_tl_logger()

    if logger is not None and mlflow_cfg.track:
        log_experiment_config_as_artifact(logger, exp_cfg.model_dump())

    try:
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
            logger=logger,
        )
    except Exception as e:
        error_msg = f"Training failed: {e}"
        logz_logger.error(error_msg)
        if logger is not None:
            logger.experiment.log_text(logger.run_id, error_msg, "training_error.txt")

    return True
