from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from stgym.config_schema import TrainConfig
from stgym.data_loader import STDataModule
from stgym.tl_model import STGymModule

mlf_logger = MLFlowLogger(
    experiment_name="lightning_logs", tracking_uri="http://127.0.0.1:8080"
)


def train(
    model: STGymModule,
    datamodule: STDataModule,
    cfg: TrainConfig,
    logger: bool = True,
    trainer_config: Optional[dict[str, any]] = None,
):
    r"""Trains a GraphGym model using PyTorch Lightning.

    Args:
        model (GraphGymModule): The GraphGym model.
        datamodule (GraphGymDataModule): The GraphGym data module.
        logger (bool, optional): Whether to enable logging during training.
            (default: :obj:`True`)
        trainer_config (dict, optional): Additional trainer configuration.
    """
    # warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

    callbacks = [EarlyStopping(monitor="val_loss", mode="min")]
    # if logger:
    #     callbacks.append(LoggerCallback())
    # if cfg.train.enable_ckpt:
    #     ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
    #     callbacks.append(ckpt_cbk)

    trainer_config = trainer_config or {}
    trainer = pl.Trainer(
        **trainer_config,
        # enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        # default_root_dir=cfg.out_dir,
        max_epochs=cfg.max_epoch,
        # accelerator=cfg.accelerator,
        devices=1,
        # 'mps' not supporting some sparse operations, therefore shouldn't be used
        accelerator="cpu" if not torch.cuda.is_available() else "gpu",
        logger=mlf_logger
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
