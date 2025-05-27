import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pydash as _
import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from torch_geometric.data import Data

from stgym.config_schema import (
    ClusteringModelConfig,
    GraphClassifierModelConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.loss import compute_classification_loss
from stgym.model import STClusteringModel, STGraphClassifier, STNodeClassifier
from stgym.optimizer import create_optimizer_from_cfg, create_scheduler
from stgym.utils import collapse_ptr_list, flatten_dict

# from torch_geometric.graphgym.loss import compute_loss
# from torch_geometric.graphgym.models.gnn import GNN
# from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
# from torch_geometric.graphgym.register import network_dict, register_network

# register_network('gnn', GNN)

torch.autograd.set_detect_anomaly(True)


@dataclass
class Split:
    train = "train"
    val = "validation"
    test = "test"


class STGymModule(pl.LightningModule):
    def __init__(
        self,
        dim_in,
        model_cfg: GraphClassifierModelConfig | ClusteringModelConfig,
        train_cfg: TrainConfig,
        task_cfg: TaskConfig,
        dim_out: int = None,
    ):
        super().__init__()
        self.my_hparams = model_cfg.model_dump() | train_cfg.model_dump()
        self.train_cfg = train_cfg
        self.task_cfg = task_cfg
        if task_cfg.type == "graph-classification":
            self.model = STGraphClassifier(dim_in, dim_out, model_cfg)
        elif task_cfg.type == "node-clustering":
            self.model = STClusteringModel(dim_in, model_cfg)
        elif task_cfg.type == "node-classification":
            self.model = STNodeClassifier(dim_in, dim_out, model_cfg)
        else:
            raise ValueError(f"Unsupported task type: {task_cfg.type}")
        self.val_step_outputs = []
        self.test_step_outputs = []

        # self.save_hyperparameters()  # do not use this, as it records the pydantic models directly

    def on_fit_start(self):
        # Perform any setup actions here
        hprams_flattened = flatten_dict(self.my_hparams, separator="/")
        self.logger.log_hyperparams(hprams_flattened)

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

        if self.task_cfg.type == "graph-classification":
            batch, pred_logits, layer_losses = self(batch)
            true = batch.y

            # the corner case of batch size = 1
            if pred_logits.ndim == 0:
                pred_logits = pred_logits.unsqueeze(-1)

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
        elif self.task_cfg.type == "node-clustering":
            ptr = batch.ptr  # used to determine instance boundaries
            batch, pred, layer_losses = self(batch)
            clustering_related_loss = sum(layer_losses[-1].values())
            step_end_time = time.time()
            if split == Split.train:
                return dict(
                    loss=clustering_related_loss,
                    step_end_time=step_end_time,
                )
            else:
                true = batch.y
                return dict(
                    ptr=ptr.detach(),
                    loss=clustering_related_loss,
                    true=true,
                    pred_score=pred.detach(),
                    step_end_time=step_end_time,
                )
        elif self.task_cfg.type == "node-classification":
            batch, pred_logits, layer_losses = self(batch)
            true = batch.y
            loss, pred_score = compute_classification_loss(
                pred_logits, true.type(torch.long)
            )

            if len(layer_losses):
                loss += sum(layer_losses[-1].values())

            step_end_time = time.time()
            return dict(
                loss=loss,
                true=true,
                pred_score=pred_score.detach(),
                step_end_time=step_end_time,
            )
        else:
            raise NotImplementedError

    def training_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split=Split.train)
        self.log("train_loss", output["loss"], prog_bar=True)
        return output

    def validation_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split=Split.val)
        self.val_step_outputs.append(output)
        self.log("val_loss", output["loss"], prog_bar=True)
        return output

    def test_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split=Split.test)
        self.test_step_outputs.append(output)
        return output

    def _extract_pred_and_test_from_step_outputs(self, split: str):
        """extract the prediction and ground truth accumulated from validation steps"""
        if split == "val":
            outputs = self.val_step_outputs
        elif split == "test":
            outputs = self.test_step_outputs
        else:
            raise ValueError(split)

        true = torch.cat([output["true"].cpu() for output in outputs])
        pred = torch.cat([output["pred_score"].cpu() for output in outputs])
        if "ptr" not in outputs[0]:
            return true, pred
        else:
            ptr = collapse_ptr_list(_.map_(outputs, lambda x: x["ptr"].cpu()))
            return true, pred, ptr

    def _shared_epoch_end(self, split: Split):

        if self.task_cfg.type == "graph-classification":
            true, pred = self._extract_pred_and_test_from_step_outputs(split=split)
            pr_auc = roc_auc_score(true, pred)
            self.log(f"{split}_pr_auc", pr_auc, prog_bar=True)
        elif self.task_cfg.type == "node-clustering":
            true, pred, ptr_batch = self._extract_pred_and_test_from_step_outputs(
                split=split
            )
            nmi_scores = []
            for start, end in zip(ptr_batch[:-1], ptr_batch[1:]):
                nmi_scores.append(
                    normalized_mutual_info_score(
                        true[start:end], pred[start:end, :].argmax(axis=1)
                    )
                )
            self.log(f"{split}_nmi", np.mean(nmi_scores), prog_bar=True)
        elif self.task_cfg.type == "node-classification":
            true, pred = self._extract_pred_and_test_from_step_outputs(split=split)
            acc = accuracy_score(true, pred.argmax(axis=1))
            micro_f1_score = f1_score(true, pred.argmax(axis=1), average="micro")
            self.log(f"{split}_accuracy", acc, prog_bar=True)
            self.log(f"{split}_micro_f1_score", micro_f1_score, prog_bar=True)

        getattr(self, f"{split}_step_outputs").clear()

    def on_validation_epoch_end(self):
        return self._shared_epoch_end("val")

    def on_test_epoch_end(self):
        return self._shared_epoch_end("test")

    def lr_scheduler_step(self, *args, **kwargs):
        # Needed for PyTorch 2.0 since the base class of LR schedulers changed.
        # TODO Remove once we only want to support PyTorch Lightning >= 2.0.
        return super().lr_scheduler_step(*args, **kwargs)
