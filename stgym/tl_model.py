import time
import pytorch_lightning as pl
from typing import Any, Dict, Tuple
from torch_geometric.data import Data
import torch

from stgym.config_schema import ModelConfig, TrainConfig
from stgym.model import STGraphClassifier
from stgym.optimizer import create_optimizer_from_cfg, create_scheduler
from stgym.loss import compute_classification_loss

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
        print(self.model)
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
        print("pred: {}".format(pred_logits))
        print("true: {}".format(true))
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
        return self._shared_step(batch, split="train")

    def validation_step(self, batch: Data, *args, **kwargs):
        return self._shared_step(batch, split="val")

    def test_step(self, batch: Data, *args, **kwargs):
        return self._shared_step(batch, split="test")

    # @property
    # def encoder(self) -> torch.nn.Module:
    #     return self.model.encoder

    # @property
    # def mp(self) -> torch.nn.Module:
    #     return self.model.mp

    # @property
    # def post_mp(self) -> torch.nn.Module:
    #     return self.model.post_mp

    # @property
    # def pre_mp(self) -> torch.nn.Module:
    #     return self.model.pre_mp

    def lr_scheduler_step(self, *args, **kwargs):
        # Needed for PyTorch 2.0 since the base class of LR schedulers changed.
        # TODO Remove once we only want to support PyTorch Lightning >= 2.0.
        return super().lr_scheduler_step(*args, **kwargs)
