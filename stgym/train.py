from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from stgym.config_schema import MLFlowConfig, TrainConfig
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule


def train(
    model: STGymModule,
    datamodule: STDataModule,
    train_cfg: TrainConfig,
    mlflow_config: MLFlowConfig,
    logger: bool = True,
    tl_train_config: Optional[dict[str, any]] = None,
):
    r"""Trains a GraphGym model using PyTorch Lightning.

    Args:
        model (GraphGymModule): The GraphGym model.
        datamodule (GraphGymDataModule): The GraphGym data module.
        logger (bool, optional): Whether to enable logging during training.
            (default: :obj:`True`)
        tl_train_config (dict, optional): Additional configuration to tl.Trainer
    """
    print(f"model device: {model.device}")
    # warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')
    logger = (
        MLFlowLogger(
            run_name=mlflow_config.run_name,
            experiment_name=mlflow_config.experiment_name,
            tracking_uri=str(mlflow_config.tracking_uri),
            tags=mlflow_config.tags,
        )
        if mlflow_config.track
        else None
    )

    callbacks = []
    if train_cfg.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=train_cfg.early_stopping.metric,
                mode=train_cfg.early_stopping.mode,
            )
        )
    # if logger:
    #     callbacks.append(LoggerCallback())
    # if cfg.train.enable_ckpt:
    #     ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
    #     callbacks.append(ckpt_cbk)

    trainer_config = tl_train_config or {}
    # Extract devices from trainer_config, fallback to 1
    devices = trainer_config.pop("devices", 1)
    trainer = pl.Trainer(
        **trainer_config,
        # enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        # default_root_dir=cfg.out_dir,
        max_epochs=train_cfg.max_epoch,
        devices=devices,  # use 'auto' in ray
        # 'mps' not supporting some sparse operations, therefore shouldn't be used
        accelerator="cpu" if not torch.cuda.is_available() else "gpu",
        logger=logger,
    )

    # make a forward pass to initialize the model
    # this is needed for DDP mode
    # for batch in datamodule.train_dataloader():
    #     print("batch.x.device", batch.x.device)
    #     model(batch)
    #     break
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
