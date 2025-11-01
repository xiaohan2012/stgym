import pytorch_lightning as pl
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning.loggers import MLFlowLogger


class MLFlowSystemMonitorCallback(pl.Callback):
    # Copied from https://github.com/Lightning-AI/pytorch-lightning/issues/20563
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            raise Exception("MLFlowSystemMonitorCallback requires MLFlowLogger")

        self.system_monitor = SystemMetricsMonitor(
            run_id=trainer.logger.run_id,
        )
        self.system_monitor.start()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.system_monitor.finish()
