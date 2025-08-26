from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from stgym.callbacks import MLFlowSystemMonitorCallback
from stgym.config_schema import MLFlowConfig, TrainConfig
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule


def train(
    model: STGymModule,
    datamodule: STDataModule,
    train_cfg: TrainConfig,
    mlflow_config: MLFlowConfig,
    tl_train_config: Optional[dict[str, any]] = None,
    logger: Optional = None,
):
    r"""Trains a GraphGym model using PyTorch Lightning.

    Args:
        model (GraphGymModule): The GraphGym model.
        datamodule (GraphGymDataModule): The GraphGym data module.
        logger (bool, optional): Whether to enable logging during training.
            (default: :obj:`True`)
        tl_train_config (dict, optional): Additional configuration to tl.Trainer
    """
    # warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

    # Configure float32 matmul precision when GPU is available
    use_gpu = torch.cuda.is_available()
    if use_gpu and train_cfg.enable_float32_matmul_precision:
        torch.set_float32_matmul_precision("medium")

    callbacks = []

    # Only add MLFlow system monitor if tracking is enabled and logger is available
    if mlflow_config.track and logger is not None:
        callbacks.append(MLFlowSystemMonitorCallback())

    if train_cfg.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=train_cfg.early_stopping.metric,
                mode=train_cfg.early_stopping.mode,
                patience=train_cfg.early_stopping.patience,
            )
        )
    # if logger:
    #     callbacks.append(LoggerCallback())
    # if cfg.train.enable_ckpt:
    #     ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
    #     callbacks.append(ckpt_cbk)

    trainer_config = tl_train_config or {}
    trainer = pl.Trainer(
        **trainer_config,
        # enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        # default_root_dir=cfg.out_dir,
        max_epochs=train_cfg.max_epoch,
        devices=train_cfg.devices,
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
