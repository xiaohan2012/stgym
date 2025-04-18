import time
import numpy as np
import pytorch_lightning as pl
from typing import Any, Dict, Tuple
from torch_geometric.data import Data
import torch

from stgym.config_schema import ModelConfig, TrainConfig
from stgym.model import STGraphClassifier
from stgym.optimizer import create_optimizer_from_cfg, create_scheduler
from stgym.loss import compute_classification_loss
from sklearn.metrics import roc_auc_score

# from torch_geometric.graphgym.loss import compute_loss
# from torch_geometric.graphgym.models.gnn import GNN
# from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
# from torch_geometric.graphgym.register import network_dict, register_network

# register_network('gnn', GNN)

torch.autograd.set_detect_anomaly(True)


class STGymModule(pl.LightningModule):
    def __init__(self, dim_in, dim_out, model_cfg: ModelConfig, train_cfg: TrainConfig):
        super().__init__()
        self.train_cfg = train_cfg
        self.model = STGraphClassifier(dim_in, dim_out, model_cfg)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer_from_cfg(self.train_cfg.optim)(
            self.model.parameters()
        )
        # optimizer = create_optimizer(self.model.parameters(), self.train_cfg.optim)
        scheduler = create_scheduler(optimizer, self.train_cfg.lr_schedule)
        return [optimizer], [scheduler]

    def _shared_step(self, batch: Data, split: str) -> Dict:
        batch.split = split

        batch, pred_logits, layer_losses = self(batch)
        true = batch.y

        # the corner case of batch size = 1
        if pred_logits.ndim == 0:
            pred_logits = pred_logits.unsqueeze(-1)

        # print("pred: {}".format(pred_logits))
        # print("true: {}".format(true))
        loss, pred_score = compute_classification_loss(pred_logits, true)

        # TODO: may attach different weights to the loss terms
        pooling_loss = sum(layer_losses[-1].values())
        step_end_time = time.time()
        return dict(
            loss=loss + pooling_loss,
            true=true,
            pred_score=pred_score.detach(),
            step_end_time=step_end_time,
        )

    def training_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split="train")
        self.log("train_loss", output["loss"], prog_bar=True)
        return output

    def validation_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split="val")
        self.validation_step_outputs.append(output)
        self.log("val_loss", output["loss"], prog_bar=True)
        return output

    def test_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split="test")
        self.test_step_outputs.append(output)
        return output

    def _extract_pred_and_test_from_step_outputs(self, split: str):
        """extract the prediction and ground truth accumulated from validation steps"""
        if split == "val":
            outputs = self.validation_step_outputs
        elif split == "test":
            outputs = self.test_step_outputs
        else:
            raise ValueError(split)

        true = torch.cat([output["true"].cpu() for output in outputs])
        pred = torch.cat([output["pred_score"].cpu() for output in outputs])
        return true, pred

    def on_validation_epoch_end(self):
        true, pred = self._extract_pred_and_test_from_step_outputs(split="val")
        pr_auc = roc_auc_score(true, pred)
        self.log("val_pr_auc", pr_auc, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        true, pred = self._extract_pred_and_test_from_step_outputs(split="test")
        pr_auc = roc_auc_score(true, pred)
        self.log("test_pr_auc", pr_auc)
        self.test_step_outputs.clear()  # free memory

    def lr_scheduler_step(self, *args, **kwargs):
        # Needed for PyTorch 2.0 since the base class of LR schedulers changed.
        # TODO Remove once we only want to support PyTorch Lightning >= 2.0.
        return super().lr_scheduler_step(*args, **kwargs)
